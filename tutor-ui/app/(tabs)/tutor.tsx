import React, { useMemo, useRef, useState } from "react";
import {
  SafeAreaView, View, Text, TextInput, TouchableOpacity, ScrollView,
  KeyboardAvoidingView, Platform, ActivityIndicator, Alert
} from "react-native";

import * as DocumentPicker from "expo-document-picker";

// IMPORTANT: Point this to your FastAPI server.
// iOS Simulator: http://localhost:8000
// Android Emulator: http://10.0.2.2:8000
// Physical device via Expo Go: http://<your-laptop-LAN-IP>:8000
const BASE_URL = "http://127.0.0.1:8000";

/** -------------------------
 * Types (align with new backend)
 * ------------------------*/

interface Exercise {
  type: "roleplay";
  instructions?: string | null;
  prompt?: string | null;
  choices?: string[] | null; // backend always null, kept for safety
  answer?: string | null;    // rubric (OOC), shown as hints when combining
  skills?: string[];
  deadline_sec?: number | null;
  branch?: "probe_deeper" | "raise_time_pressure" | "de_escalate" | "switch_perspective" | "escalate_objection" | "close_out" | null;
}

interface LessonPlan {
  chapter_summary: string;
  learning_objectives: string[];
  roleplay_persona: string;
  first_exercise?: Exercise; // from /ingest
}

interface IngestResponse {
  user_id: string;
  lesson_plan: LessonPlan;
  first_prompt: string;
}

interface ChatResponse {
  user_id: string;
  reply: string; // may sometimes contain JSON as string; we sanitize
  state: {
    chapter_title: string;
    chapter_source?: string | null;
    chapter_summary: string;
    learning_objectives: string[];
    roleplay_persona: string;
    signals_preview: { empathy: number; text_evidence: number; bias_check: number; curiosity: number };
    exercise_queue_len: number;
  };
  new_exercise?: Exercise | null;
  citations?: string[] | null;
}

interface EndResponse { user_id: string; final_feedback: {
  summary: string;
  strengths?: string[];
  growth?: string[];
  chapter_evidence?: { quote_or_paraphrase: string; why_it_matters: string; citations?: string[] }[];
  metrics?: { empathy: number; text_evidence: number; bias_check: number; curiosity: number };
}; }

/** -------------------------
 * Small UI atoms
 * ------------------------*/

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <View style={{ marginVertical: 12 }}>
    <Text style={{ fontWeight: "700", fontSize: 16, marginBottom: 6 }}>{title}</Text>
    <View style={{ gap: 8 }}>{children}</View>
  </View>
);

const Input: React.FC<{
  label: string;
  value: string;
  onChangeText: (t: string) => void;
  multiline?: boolean;
  placeholder?: string;
}> = ({ label, value, onChangeText, multiline, placeholder }) => (
  <View style={{ marginBottom: 10 }}>
    <Text style={{ fontWeight: "600", marginBottom: 6 }}>{label}</Text>
    <TextInput
      value={value}
      onChangeText={onChangeText}
      multiline={multiline}
      placeholder={placeholder}
      placeholderTextColor="#999"
      style={{
        borderWidth: 1, borderColor: "#ccc", borderRadius: 8, padding: 10,
        minHeight: multiline ? 80 : undefined,
      }}
    />
  </View>
);

const Button: React.FC<{ title: string; onPress: () => void; disabled?: boolean }> = ({ title, onPress, disabled }) => (
  <TouchableOpacity
    onPress={onPress}
    disabled={disabled}
    style={{
      backgroundColor: disabled ? "#bbb" : "#111827",
      paddingVertical: 12, paddingHorizontal: 16, borderRadius: 10, alignItems: "center",
    }}
  >
    <Text style={{ color: "white", fontWeight: "700" }}>{title}</Text>
  </TouchableOpacity>
);

const Badge: React.FC<{ label: string }> = ({ label }) => (
  <View style={{ backgroundColor: "#E5E7EB", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 9999, marginRight: 6, marginBottom: 6 }}>
    <Text style={{ fontSize: 12 }}>{label}</Text>
  </View>
);

const Bubble: React.FC<{ role: "user" | "assistant"; text: string }> = ({ role, text }) => (
  <View style={{
    alignSelf: role === "user" ? "flex-end" : "flex-start",
    backgroundColor: role === "user" ? "#111827" : "#F3F4F6",
    padding: 10, marginVertical: 6, borderRadius: 14, maxWidth: "85%",
  }}>
    <Text style={{ color: role === "user" ? "#fff" : "#111827" }}>{text}</Text>
  </View>
);

const CitationsRow: React.FC<{ cites?: string[] | null }> = ({ cites }) =>
  !cites || cites.length === 0 ? null : (
    <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
      {cites.map((c, i) => <Badge key={i} label={c} />)}
    </View>
  );

/** -------------------------
 * Sanitizers & Composers
 * ------------------------*/
function stripFences(t: string): string {
  if (!t) return t;
  const s = t.trim();
  if (s.startsWith("```")) {
    return s.split("\n").filter((ln) => !ln.trim().startsWith("```")) .join("\n").trim();
  }
  return s;
}

function safeReply(raw: any): string {
  if (raw == null) return "";
  if (typeof raw === "string") {
    const s = stripFences(raw);
    if (s.startsWith("{") && s.endsWith("}")) {
      try {
        const obj = JSON.parse(s);
        if (obj && typeof obj.reply === "string") return obj.reply;
      } catch {}
    }
    const m = s.match(/"reply"\s*:\s*"([\s\S]*?)"\s*(,|\})/);
    if (m) {
      try { return JSON.parse(`"${m[1]}"`); } catch { return m[1]; }
    }
    return s;
  }
  if (typeof raw === "object" && typeof raw.reply === "string") return raw.reply;
  return String(raw);
}

function combineFirstBotBubble(lp: LessonPlan, firstPrompt: string): string {
  const persona = lp.roleplay_persona ? `🎭 ${lp.roleplay_persona}` : "";
  const ex = (lp.first_exercise || {}) as Exercise;
  const instr = ex.instructions ? `\n\n📝 Task: ${ex.instructions}` : "";
  const prompt = firstPrompt || ex.prompt || "";
  const clock = typeof ex.deadline_sec === "number" ? `  ⏳${ex.deadline_sec}s` : "";
  const hint = ex.answer ? `\n\n💡 Hints: ${ex.answer}` : "";
  const skills = ex.skills && ex.skills.length ? `\n\n🎯 Skills: ${ex.skills.join(", ")}` : "";
  const body = prompt ? `\n\n${prompt}${clock}` : "";
  return [persona, instr, body, hint, skills].filter(Boolean).join("");
}

function combineChatAssistantBubble(reply: string, ex?: Exercise | null): string {
  const r = safeReply(reply);
  if (!ex) return r;
  const instr = ex.instructions ? `\n\n📝 Next: ${ex.instructions}` : "";
  const p = ex.prompt ? `\n${ex.prompt}` : "";
  const clock = typeof ex.deadline_sec === "number" ? `  ⏳${ex.deadline_sec}s` : "";
  const hint = ex.answer ? `\n\n💡 Hints: ${ex.answer}` : "";
  const skills = ex.skills && ex.skills.length ? `\n\n🎯 Skills: ${ex.skills.join(", ")}` : "";
  return [r, instr, p + clock, hint, skills].filter(Boolean).join("");
}

/** -------------------------
 * API helpers
 * ------------------------*/
async function toFormDataFile(
  picked: { uri: string; name?: string; mime?: string }
): Promise<any /* RN file or Web File */> {
  const filename = picked.name || "chapter.txt";
  const mime = picked.mime || "text/plain";

  if (Platform.OS === "web") {
    const blob = await fetch(picked.uri).then(r => r.blob());
    return new File([blob], filename, { type: mime });
  }
  return { uri: picked.uri, name: filename, type: mime } as any;
}

async function apiIngestFile(fd: FormData): Promise<IngestResponse> {
  const res = await fetch(`${BASE_URL}/ingest_file`, { method: "POST", body: fd });
  const body = await res.text();
  if (!res.ok) throw new Error(`Ingest (file) failed: ${res.status} ${body}`);
  return JSON.parse(body);
}

async function apiIngest(payload: {
  user_id: string;
  chapter_title: string;
  chapter_text: string;
  chapter_source?: string | null;
  profile: { background: string; goals: string; level?: string | null };
}): Promise<IngestResponse> {
  const res = await fetch(`${BASE_URL}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Ingest failed: ${res.status} ${t}`);
  }
  return res.json();
}

async function apiChat(payload: { user_id: string; message: string }): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Chat failed: ${res.status} ${t}`);
  }
  return res.json();
}

async function apiEnd(payload: { user_id: string }): Promise<EndResponse> {
  const res = await fetch(`${BASE_URL}/end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`End failed: ${res.status} ${t}`);
  }
  return res.json();
}

/** -------------------------
 * Screen Component
 * ------------------------*/

type Msg = { kind: "bubble"; role: "user" | "assistant"; text: string; citations?: string[] | null };

export default function TutorScreen() {
  const [screen, setScreen] = useState<"setup" | "chat">("setup");
  const [pickedFile, setPickedFile] = useState<{ uri: string; name: string; mime: string } | null>(null);

  // Setup form — ONLY what's required by backend
  const [userId, setUserId] = useState("demo-user");
  const [chapterTitle, setChapterTitle] = useState("");
  const [chapterText, setChapterText] = useState("");
  const [background, setBackground] = useState("");
  const [goals, setGoals] = useState("");
  const [level, setLevel] = useState("beginner");

  // Lesson + Chat state
  const [lesson, setLesson] = useState<LessonPlan | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ended, setEnded] = useState(false);
  const [finalFeedback, setFinalFeedback] = useState<EndResponse["final_feedback"] | null>(null);
  const scrollRef = useRef<ScrollView>(null);

  const header = useMemo(() => {
    if (!lesson) return null;
    return (
      <View style={{ borderBottomWidth: 1, borderColor: "#E5E7EB", paddingBottom: 8, marginBottom: 8 }}>
        <Text style={{ fontWeight: "800", fontSize: 18, marginBottom: 6 }}>{chapterTitle || "Chapter"}</Text>
        <Text style={{ color: "#374151", marginBottom: 6 }}>{lesson.chapter_summary}</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginTop: 4 }}>
          {lesson.learning_objectives.map((o, i) => (<Badge key={i} label={o} />))}
        </ScrollView>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
          <Badge label={`Persona: ${lesson.roleplay_persona.slice(0, 40)}${lesson.roleplay_persona.length > 40 ? "…" : ""}`} />
        </View>
      </View>
    );
  }, [lesson, chapterTitle]);

  const startLesson = async () => {
    try {
      if (!userId || !(pickedFile || chapterText) || !background || !goals) {
        Alert.alert("Missing info", "Please provide User ID, chapter (file or text), background, and goals.");
        return;
      }
      setLoading(true);
      let res: IngestResponse;

      if (pickedFile) {
        const fd = new FormData();
        fd.append("user_id", userId);
        fd.append("chapter_title", chapterTitle || ""); // backend will auto-title if empty
        fd.append("profile_background", background);
        fd.append("profile_goals", goals);
        fd.append("profile_level", level || "");
        const filePart = await toFormDataFile(pickedFile);
        fd.append("chapter_file", filePart);
        res = await apiIngestFile(fd);
      } else {
        res = await apiIngest({
          user_id: userId,
          chapter_title: chapterTitle || "",
          chapter_text: chapterText,
          profile: { background, goals, level },
        });
      }

      const lp: LessonPlan = res.lesson_plan;
      setLesson(lp);

      const combined = combineFirstBotBubble(lp, res.first_prompt);
      const initMsgs: Msg[] = [{ kind: "bubble", role: "assistant", text: combined }];
      setMessages(initMsgs);
      setScreen("chat");
    } catch (e: any) {
      Alert.alert("Error", e?.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (text: string) => {
    if (ended) return; // lock after debrief
    if (!text.trim()) return;
    const userBubble: Msg = { kind: "bubble", role: "user", text };
    setMessages((m) => [...m, userBubble]);
    setInput("");

    try {
      const res = await apiChat({ user_id: userId, message: text });
      const combined = combineChatAssistantBubble(res.reply, res.new_exercise || undefined);
      const next: Msg[] = [
        { kind: "bubble", role: "assistant", text: combined, citations: res.citations || null },
      ];
      setMessages((m) => [...m, ...next]);
    } catch (e: any) {
      Alert.alert("Error", e?.message || String(e));
    }
  };

  const callEnd = async () => {
    if (ended) return;
    try {
      setLoading(true);
      const res = await apiEnd({ user_id: userId });
      setFinalFeedback(res.final_feedback);
      setEnded(true);
    } catch (e: any) {
      Alert.alert("Error", e?.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  // ---------------
  // Render
  // ---------------
  if (screen === "setup") {
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: "#fff" }}>
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ flex: 1 }}>
          <ScrollView contentContainerStyle={{ padding: 16 }}>
            <Text style={{ fontSize: 22, fontWeight: "800", marginBottom: 12 }}>Start a Scene Reenactment</Text>

            <Section title="User & Chapter (Required)">
              <Input label="User ID" value={userId} onChangeText={setUserId} />
              <Input label="Chapter Title (optional)" value={chapterTitle} onChangeText={setChapterTitle} />

              <TouchableOpacity
                onPress={async () => {
                  const res = await DocumentPicker.getDocumentAsync({
                    type: "text/plain",
                    multiple: false,
                    copyToCacheDirectory: true,
                  });
                  if (res.canceled) return;
                  const file = res.assets[0];
                  setPickedFile({ uri: file.uri, name: file.name ?? "chapter.txt", mime: file.mimeType ?? "text/plain" });
                }}
                style={{ paddingVertical: 10, paddingHorizontal: 12, borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 8 }}
              >
                <Text>{pickedFile ? `📄 ${pickedFile.name}` : "Choose .txt file (or paste text below)"}</Text>
              </TouchableOpacity>

              <Input label="OR Paste Chapter Text" value={chapterText} onChangeText={setChapterText} multiline placeholder="Paste the chapter/material here" />
            </Section>

            <Section title="Learner Profile (Required)">
              <Input label="Background" value={background} onChangeText={setBackground} />
              <Input label="Goal" value={goals} onChangeText={setGoals} />
              <Input label="Level (optional)" value={level} onChangeText={setLevel} placeholder="beginner / intermediate / advanced" />
            </Section>

            <Button title={loading ? "Starting…" : "Start"} onPress={startLesson} disabled={loading} />
            {loading ? <View style={{ marginTop: 12 }}><ActivityIndicator /></View> : null}
            <View style={{ height: 30 }} />
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  // Chat screen — minimal: summary header + debrief (optional) + chat list + composer
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#fff" }}>
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ flex: 1 }}>
        <View style={{ flex: 1, paddingHorizontal: 12, paddingTop: 8 }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
            <View style={{ flex: 1 }}>{header}</View>
            {!ended ? (
              <TouchableOpacity onPress={callEnd} style={{ paddingVertical: 8, paddingHorizontal: 12, borderWidth: 1, borderColor: '#D1D5DB', borderRadius: 8 }}>
                <Text>End & Debrief</Text>
              </TouchableOpacity>
            ) : null}
          </View>

          {/* Debrief card */}
          {finalFeedback ? (
            <View style={{ borderWidth: 1, borderColor: '#E5E7EB', borderRadius: 12, padding: 12, marginBottom: 8, backgroundColor: '#F9FAFB' }}>
              <Text style={{ fontWeight: '800', marginBottom: 6 }}>Session Debrief</Text>
              <Text style={{ marginBottom: 8 }}>{finalFeedback.summary}</Text>
              {finalFeedback.strengths && finalFeedback.strengths.length ? (
                <View style={{ marginBottom: 8 }}>
                  <Text style={{ fontWeight: '700', marginBottom: 4 }}>Strengths</Text>
                  <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
                    {finalFeedback.strengths.map((s, i) => <Badge key={i} label={s} />)}
                  </View>
                </View>
              ) : null}
              {finalFeedback.growth && finalFeedback.growth.length ? (
                <View style={{ marginBottom: 8 }}>
                  <Text style={{ fontWeight: '700', marginBottom: 4 }}>Next steps</Text>
                  {finalFeedback.growth.map((g, i) => <Text key={i}>• {g}</Text>)}
                </View>
              ) : null}
              {finalFeedback.chapter_evidence && finalFeedback.chapter_evidence.length ? (
                <View>
                  <Text style={{ fontWeight: '700', marginBottom: 4 }}>Evidence</Text>
                  {finalFeedback.chapter_evidence.map((e, i) => (
                    <View key={i} style={{ marginBottom: 6 }}>
                      <Text style={{ fontStyle: 'italic' }}>
                        “{e.quote_or_paraphrase}”
                      </Text>
                      <Text>{e.why_it_matters}</Text>
                      {e.citations && e.citations.length ? <CitationsRow cites={e.citations} /> : null}
                    </View>
                  ))}
                </View>
              ) : null}
            </View>
          ) : null}

          <ScrollView
            ref={scrollRef}
            onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
            contentContainerStyle={{ paddingBottom: 8 }}
          >
            {messages.map((m, idx) => (
              <View key={idx} style={{ marginBottom: 2 }}>
                <Bubble role={m.role} text={m.text} />
                {m.role === "assistant" ? <CitationsRow cites={m.citations} /> : null}
              </View>
            ))}
          </ScrollView>

          <View style={{ flexDirection: "row", alignItems: "center", gap: 8, paddingVertical: 8 }}>
            <TextInput
              value={input}
              onChangeText={setInput}
              editable={!ended}
              placeholder={ended ? "Session ended — view debrief above" : "Type your line in the scene…"}
              placeholderTextColor="#999"
              style={{ flex: 1, borderWidth: 1, borderColor: "#E5E7EB", borderRadius: 9999, paddingHorizontal: 14, paddingVertical: 10 }}
            />
            <TouchableOpacity disabled={ended} onPress={() => sendMessage(input)} style={{ backgroundColor: ended ? "#9CA3AF" : "#111827", paddingVertical: 12, paddingHorizontal: 16, borderRadius: 9999 }}>
              <Text style={{ color: "#fff", fontWeight: "700" }}>Send</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
