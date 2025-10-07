import React, { useEffect, useMemo, useRef, useState } from "react";
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
 * Types (align with backend)
 * ------------------------*/

type ExerciseType = "mcq" | "short_answer" | "coding" | "roleplay" | "reflection";

interface Exercise {
  type: ExerciseType;
  instructions?: string | null;
  prompt?: string | null;
  choices?: string[] | null;
  answer?: string | null; // rubric (OOC)
  skills?: string[];
  deadline_sec?: number | null;
  branch?: "probe_deeper" | "raise_time_pressure" | "de_escalate" | "switch_perspective" | null;
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
  reply: string;
  state: {
    chapter_title: string;
    chapter_source?: string | null;
    chapter_summary: string;
    learning_objectives: string[];
    roleplay_persona: string;
    progress: { mastery: number; correct_in_a_row: number; attempts: number };
    exercise_queue_len: number;
  };
  new_exercise?: Exercise | null;
  assessment?: { correct: boolean | null; explanation?: string; delta_mastery?: number } | null;
  citations?: string[] | null;
}

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
    backgroundColor: role === "user" ? "#2563EB" : "#F3F4F6",
    padding: 10, marginVertical: 6, borderRadius: 14, maxWidth: "85%",
  }}>
    <Text style={{ color: role === "user" ? "#fff" : "#111827" }}>{text}</Text>
  </View>
);

const CoachNote: React.FC<{ text?: string }> = ({ text }) =>
  !text ? null : (
    <View style={{ backgroundColor: "#FFFBEB", borderColor: "#F59E0B", borderWidth: 1, padding: 8, borderRadius: 10, marginTop: 6 }}>
      <Text style={{ color: "#92400E" }}>Coach: {text}</Text>
    </View>
  );

const CitationsRow: React.FC<{ cites?: string[] | null }> = ({ cites }) =>
  !cites || cites.length === 0 ? null : (
    <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
      {cites.map((c, i) => <Badge key={i} label={c} />)}
    </View>
  );

const ExerciseCard: React.FC<{
  ex: Exercise;
  onAnswer?: (text: string) => void;
  onCountdownDone?: () => void;
}> = ({ ex, onAnswer, onCountdownDone }) => {
  const [val, setVal] = useState("");
  const [remaining, setRemaining] = useState<number | null>(
    typeof ex.deadline_sec === "number" ? ex.deadline_sec : null
  );

  useEffect(() => {
    if (remaining === null) return;
    if (remaining <= 0) {
      onCountdownDone?.();
      return;
    }
    const t = setTimeout(() => setRemaining((s) => (s === null ? null : s - 1)), 1000);
    return () => clearTimeout(t);
  }, [remaining, onCountdownDone]);

  return (
    <View style={{ borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 12, padding: 12, marginVertical: 8 }}>
      <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
        <Text style={{ fontWeight: "700" }}>{ex.instructions || "Exercise"}</Text>
        {remaining !== null && remaining >= 0 ? (
          <Badge label={`⏳ ${remaining}s`} />
        ) : null}
      </View>
      {ex.prompt ? <Text style={{ marginTop: 6, marginBottom: 10 }}>{ex.prompt}</Text> : null}

      {ex.choices && Array.isArray(ex.choices) && ex.choices.length > 0 ? (
        <View style={{ gap: 8 }}>
          {ex.choices.map((c, idx) => (
            <Button key={idx} title={c} onPress={() => onAnswer?.(c)} />
          ))}
        </View>
      ) : (
        <View>
          <TextInput
            value={val}
            onChangeText={setVal}
            placeholder="Type your answer..."
            placeholderTextColor="#999"
            multiline
            style={{ borderWidth: 1, borderColor: "#E5E7EB", borderRadius: 8, padding: 8, minHeight: 60, marginBottom: 8 }}
          />
          <Button title="Submit" onPress={() => onAnswer?.(val)} />
        </View>
      )}

      {ex.skills && ex.skills.length > 0 ? (
        <View style={{ flexDirection: "row", flexWrap: "wrap", marginTop: 10 }}>
          {ex.skills.map((s, i) => <Badge key={i} label={s} />)}
        </View>
      ) : null}
    </View>
  );
};

/** -------------------------
 * API helpers
 * ------------------------*/
async function apiIngestFile(fd: FormData): Promise<IngestResponse> {
  const res = await fetch(`${BASE_URL}/ingest_file`, {
    method: "POST",
    body: fd, // fetch sets multipart boundaries automatically
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Ingest (file) failed: ${res.status} ${t}`);
  }
  return res.json();
}


async function apiScenario(payload: {
  user_id: string;
  book: string;
  user_role: string;
  bot_role: string;
  difficulty: number;
  learning_style?: string | null;
  time_pressure?: boolean;
  emotion?: "supportive" | "neutral" | "challenging";
  roleplay_only?: boolean;
}): Promise<{ ok: boolean }> {
  const res = await fetch(`${BASE_URL}/scenario`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Scenario failed: ${res.status} ${t}`);
  }
  return res.json();
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

/** -------------------------
 * Screen Component
 * ------------------------*/

type Msg =
  | { kind: "bubble"; role: "user" | "assistant"; text: string; citations?: string[] | null; coach?: string | null }
  | { kind: "exercise"; ex: Exercise };

export default function TutorScreen() {
  const [screen, setScreen] = useState<"setup" | "chat">("setup");
  const [pickedFile, setPickedFile] = useState<{ uri: string; name: string; mime: string } | null>(null);


  // Setup form
  const [userId, setUserId] = useState("demo-user");
  const [chapterTitle, setChapterTitle] = useState("Recursion Basics");
  const [chapterText, setChapterText] = useState("Paste your chapter text here…");
  const [chapterSource, setChapterSource] = useState("");

  // Profile
  const [background, setBackground] = useState("Finance professional dabbling in Python");
  const [goals, setGoals] = useState("Crack recursion and write clean functions");
  const [level, setLevel] = useState("beginner");

  // Scenario (NEW)
  const [book, setBook] = useState("Sample Book");
  const [userRole, setUserRole] = useState("Learner");
  const [botRole, setBotRole] = useState("Mentor");
  const [difficulty, setDifficulty] = useState("2");
  const [learningStyle, setLearningStyle] = useState("Socratic");
  const [timePressure, setTimePressure] = useState(false);
  const [emotion, setEmotion] = useState<"supportive" | "neutral" | "challenging">("supportive");

  // Lesson + Chat state
  const [lesson, setLesson] = useState<LessonPlan | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<ScrollView>(null);
  const progress = useRef({ mastery: 0, streak: 0, attempts: 0 });

  const header = useMemo(() => {
    if (!lesson) return null;
    return (
      <View style={{ borderBottomWidth: 1, borderColor: "#E5E7EB", paddingBottom: 8, marginBottom: 8 }}>
        <Text style={{ fontWeight: "800", fontSize: 18, marginBottom: 6 }}>{chapterTitle}</Text>
        <Text style={{ color: "#374151", marginBottom: 6 }}>{lesson.chapter_summary}</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginTop: 4 }}>
          {lesson.learning_objectives.map((o, i) => (<Badge key={i} label={o} />))}
        </ScrollView>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
          <Badge label={`Persona: ${lesson.roleplay_persona.slice(0, 40)}${lesson.roleplay_persona.length > 40 ? "…" : ""}`} />
          <Badge label={`Mastery: ${progress.current.mastery}%`} />
          <Badge label={`Streak: ${progress.current.streak}`} />
        </View>
      </View>
    );
  }, [lesson, chapterTitle]);

  const startLesson = async () => {
  try {
    setLoading(true);
    let res: IngestResponse;

    if (pickedFile) {
      const fd = new FormData();
      fd.append("user_id", userId);
      fd.append("chapter_title", chapterTitle);
      fd.append("profile_background", background);
      fd.append("profile_goals", goals);
      fd.append("profile_level", level);
      if (chapterSource) fd.append("chapter_source", chapterSource);

      fd.append("chapter_file", {
        uri: pickedFile.uri,
        name: pickedFile.name,
        type: pickedFile.mime || "text/plain",
      } as any);

      res = await apiIngestFile(fd);
    } else {
      res = await apiIngest({
        user_id: userId,
        chapter_title: chapterTitle,
        chapter_text: chapterText,
        chapter_source: chapterSource || undefined,
        profile: { background, goals, level },
      });
    }

    const lp: LessonPlan = res.lesson_plan;
    setLesson(lp);

    const initMsgs: any[] = [];
    if (res.first_prompt) initMsgs.push({ kind: "bubble", role: "assistant", text: res.first_prompt });
    if (lp.first_exercise) initMsgs.push({ kind: "exercise", ex: lp.first_exercise });
    setMessages(initMsgs);
    setScreen("chat");
  } catch (e: any) {
    Alert.alert("Error", e?.message || String(e));
  } finally {
    setLoading(false);
  }
};


  const sendMessage = async (text: string) => {
    if (!text.trim()) return;
    const userBubble: Msg = { kind: "bubble", role: "user", text };
    setMessages((m) => [...m, userBubble]);
    setInput("");

    try {
      const res = await apiChat({ user_id: userId, message: text });

      // Update progress
      progress.current = {
        mastery: res.state.progress.mastery,
        streak: res.state.progress.correct_in_a_row,
        attempts: res.state.progress.attempts,
      };

      // Assistant bubble (with optional coaching + citations)
      setMessages((m) => [
        ...m,
        {
          kind: "bubble",
          role: "assistant",
          text: res.reply || "(No reply)",
          citations: res.citations || null,
          coach: res.assessment?.explanation || null,
        },
      ]);

      // Next exercise if any
      if (res.new_exercise) {
        setMessages((m) => [...m, { kind: "exercise", ex: res.new_exercise as Exercise }]);
      }
    } catch (e: any) {
      Alert.alert("Error", e?.message || String(e));
    }
  };

  const onAnswerExercise = (answerText: string) => {
    sendMessage(answerText);
  };

  // If an exercise times out client-side, gently nudge user to send any message to continue.
  const onExerciseTimeout = () => {
    setMessages((m) => [
      ...m,
      { kind: "bubble", role: "assistant", text: "⏰ Time’s up for that turn—reply with your next move and we’ll review briefly.", citations: null, coach: null },
    ]);
  };

  if (screen === "setup") {
    return (
      <SafeAreaView style={{ flex: 1, backgroundColor: "#fff" }}>
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ flex: 1 }}>
          <ScrollView contentContainerStyle={{ padding: 16 }}>
            <Text style={{ fontSize: 22, fontWeight: "800", marginBottom: 12 }}>Tutor Setup</Text>

            <Section title="User & Chapter">
              <Input label="User ID" value={userId} onChangeText={setUserId} />
              <Input label="Chapter Title" value={chapterTitle} onChangeText={setChapterTitle} />
              <Input label="Chapter Source (optional)" value={chapterSource} onChangeText={setChapterSource} />
              <Input label="Chapter Text" value={chapterText} onChangeText={setChapterText} multiline placeholder="Paste the chapter/material here" />
              <TouchableOpacity
                onPress={async () => {
                  const res = await DocumentPicker.getDocumentAsync({
                    type: "text/plain",
                    multiple: false,
                    copyToCacheDirectory: true,
                  });
                  if (res.canceled) return;
                  const file = res.assets[0];
                  setPickedFile({
                    uri: file.uri,
                    name: file.name ?? "chapter.txt",
                    mime: file.mimeType ?? "text/plain",
                  });
                }}
                style={{ paddingVertical: 10, paddingHorizontal: 12, borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 8 }}
              >
                <Text>{pickedFile ? `📄 ${pickedFile.name}` : "Choose .txt file"}</Text>
              </TouchableOpacity>
            </Section>

            <Section title="Profile for Personalization">
              <Input label="Background" value={background} onChangeText={setBackground} />
              <Input label="Goals" value={goals} onChangeText={setGoals} />
              <Input label="Level" value={level} onChangeText={setLevel} placeholder="beginner / intermediate / advanced" />
            </Section>

            <Section title="Scenario (Role-play)">
              <Input label="Book" value={book} onChangeText={setBook} />
              <Input label="Your Role (Learner)" value={userRole} onChangeText={setUserRole} />
              <Input label="Bot Role (Mentor/CFO/etc.)" value={botRole} onChangeText={setBotRole} />
              <Input label="Difficulty (1–5)" value={difficulty} onChangeText={setDifficulty} />
              <Input label="Learning Style" value={learningStyle} onChangeText={setLearningStyle} placeholder="Socratic / examples / practice-first" />
              <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                <TouchableOpacity onPress={() => setTimePressure((s) => !s)} style={{ paddingVertical: 8, paddingHorizontal: 12, borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 8 }}>
                  <Text>{timePressure ? "✅ Time Pressure: ON" : "⏱️ Time Pressure: OFF"}</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() =>
                    setEmotion((e) => (e === "supportive" ? "neutral" : e === "neutral" ? "challenging" : "supportive"))
                  }
                  style={{ paddingVertical: 8, paddingHorizontal: 12, borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 8 }}
                >
                  <Text>Emotion: {emotion}</Text>
                </TouchableOpacity>
              </View>
            </Section>

            <Button title={loading ? "Starting…" : "Start Lesson"} onPress={startLesson} disabled={loading} />
            {loading ? <View style={{ marginTop: 12 }}><ActivityIndicator /></View> : null}
            <View style={{ height: 30 }} />
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#fff" }}>
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ flex: 1 }}>
        <View style={{ flex: 1, paddingHorizontal: 12, paddingTop: 8 }}>
          {header}
          <ScrollView
            ref={scrollRef}
            onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
          >
            {messages.map((m, idx) => {
              if (m.kind === "bubble") {
                return (
                  <View key={idx} style={{ marginBottom: 2 }}>
                    <Bubble role={m.role} text={m.text} />
                    {m.role === "assistant" ? (
                      <>
                        <CoachNote text={m.coach || undefined} />
                        <CitationsRow cites={m.citations} />
                      </>
                    ) : null}
                  </View>
                );
              }
              // exercise
              return (
                <ExerciseCard
                  key={idx}
                  ex={m.ex}
                  onAnswer={onAnswerExercise}
                  onCountdownDone={onExerciseTimeout}
                />
              );
            })}
            <View style={{ height: 8 }} />
          </ScrollView>

          <View style={{ flexDirection: "row", alignItems: "center", gap: 8, paddingVertical: 8 }}>
            <TextInput
              value={input}
              onChangeText={setInput}
              placeholder="Type your message…"
              placeholderTextColor="#999"
              style={{ flex: 1, borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 9999, paddingHorizontal: 14, paddingVertical: 10 }}
            />
            <TouchableOpacity onPress={() => sendMessage(input)} style={{ backgroundColor: "#111827", paddingVertical: 12, paddingHorizontal: 16, borderRadius: 9999 }}>
              <Text style={{ color: "#fff", fontWeight: "700" }}>Send</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
