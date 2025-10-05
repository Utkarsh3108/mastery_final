import React, { useMemo, useRef, useState } from "react";
import { SafeAreaView, View, Text, TextInput, TouchableOpacity, ScrollView, KeyboardAvoidingView, Platform, ActivityIndicator, Alert } from "react-native";

/**
 * Expo Router screen for the Tutor Bot API
 * Place this file at: app/(tabs)/tutor.tsx
 * Then register it in app/(tabs)/_layout.tsx as a new Tab (see instructions in chat).
 */

// IMPORTANT: Point this to your FastAPI server.
// iOS Simulator: http://localhost:8000
// Android Emulator: http://10.0.2.2:8000
// Physical device via Expo Go: http://<your-laptop-LAN-IP>:8000
const BASE_URL = "http://127.0.0.1:8000";

// ---------- Types ----------

type ExerciseType = "mcq" | "short_answer" | "coding" | "roleplay" | "reflection";

interface Exercise {
  type: ExerciseType;
  instructions?: string | null;
  prompt?: string | null;
  choices?: string[] | null;
  answer?: string | null; // may be a rubric
  skills?: string[];
}

interface LessonPlan {
  chapter_summary: string;
  learning_objectives: string[];
  roleplay_persona: string;
  first_exercise?: Exercise; // from /ingest
}

interface IngestResponse {
  user_id: string;
  lesson_plan: {
    chapter_summary: string;
    learning_objectives: string[];
    roleplay_persona: string;
    first_exercise?: Exercise;
  };
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
}

// ---------- Small UI atoms ----------

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
        borderWidth: 1,
        borderColor: "#ccc",
        borderRadius: 8,
        padding: 10,
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
      paddingVertical: 12,
      paddingHorizontal: 16,
      borderRadius: 10,
      alignItems: "center",
    }}
  >
    <Text style={{ color: "white", fontWeight: "700" }}>{title}</Text>
  </TouchableOpacity>
);

const Badge: React.FC<{ label: string }> = ({ label }) => (
  <View style={{ backgroundColor: "#E5E7EB", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 9999 }}>
    <Text style={{ fontSize: 12 }}>{label}</Text>
  </View>
);

const Bubble: React.FC<{ role: "user" | "assistant"; text: string }> = ({ role, text }) => (
  <View style={{
    alignSelf: role === "user" ? "flex-end" : "flex-start",
    backgroundColor: role === "user" ? "#2563EB" : "#F3F4F6",
    padding: 10,
    marginVertical: 6,
    borderRadius: 14,
    maxWidth: "85%",
  }}>
    <Text style={{ color: role === "user" ? "#fff" : "#111827" }}>{text}</Text>
  </View>
);

const ExerciseCard: React.FC<{ ex: Exercise; onAnswer?: (text: string) => void }> = ({ ex, onAnswer }) => {
  const [val, setVal] = useState("");
  return (
    <View style={{ borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 12, padding: 12, marginVertical: 8 }}>
      {ex.instructions ? <Text style={{ fontWeight: "700", marginBottom: 6 }}>{ex.instructions}</Text> : null}
      {ex.prompt ? <Text style={{ marginBottom: 10 }}>{ex.prompt}</Text> : null}

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
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
          {ex.skills.map((s, i) => (
            <Badge key={i} label={s} />
          ))}
        </View>
      ) : null}
    </View>
  );
};

// ---------- API helpers ----------

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

// ---------- Screen Component ----------

export default function TutorScreen() {
  const [screen, setScreen] = useState<"setup" | "chat">("setup");

  // Setup form
  const [userId, setUserId] = useState("demo-user");
  const [chapterTitle, setChapterTitle] = useState("Recursion Basics");
  const [chapterText, setChapterText] = useState("Paste your chapter text here…");
  const [chapterSource, setChapterSource] = useState("");
  const [background, setBackground] = useState("Finance professional dabbling in Python");
  const [goals, setGoals] = useState("Crack recursion and write clean functions");
  const [level, setLevel] = useState("beginner");

  // Lesson + Chat state
  const [lesson, setLesson] = useState<LessonPlan | null>(null);
  const [messages, setMessages] = useState<Array<{ kind: "bubble"; role: "user" | "assistant"; text: string } | { kind: "exercise"; ex: Exercise }>>([]);
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
          {lesson.learning_objectives.map((o, i) => (
            <Badge key={i} label={o} />
          ))}
        </ScrollView>
        <View style={{ flexDirection: "row", gap: 8, marginTop: 8 }}>
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
      const res = await apiIngest({
        user_id: userId,
        chapter_title: chapterTitle,
        chapter_text: chapterText,
        chapter_source: chapterSource || undefined,
        profile: { background, goals, level },
      });

      const lp: LessonPlan = {
        chapter_summary: res.lesson_plan.chapter_summary,
        learning_objectives: res.lesson_plan.learning_objectives,
        roleplay_persona: res.lesson_plan.roleplay_persona,
        first_exercise: res.lesson_plan.first_exercise,
      };
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
    const userBubble = { kind: "bubble" as const, role: "user" as const, text };
    setMessages((m) => [...m, userBubble]);
    setInput("");

    try {
      const res = await apiChat({ user_id: userId, message: text });
      progress.current = {
        mastery: res.state.progress.mastery,
        streak: res.state.progress.correct_in_a_row,
        attempts: res.state.progress.attempts,
      };
      setMessages((m) => [...m, { kind: "bubble", role: "assistant", text: res.reply }]);
      if (res.new_exercise) setMessages((m) => [...m, { kind: "exercise", ex: res.new_exercise as Exercise }]);
    } catch (e: any) {
      Alert.alert("Error", e?.message || String(e));
    }
  };

  const onAnswerExercise = (answerText: string) => {
    sendMessage(answerText);
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
            </Section>
            <Section title="Profile for Personalization">
              <Input label="Background" value={background} onChangeText={setBackground} />
              <Input label="Goals" value={goals} onChangeText={setGoals} />
              <Input label="Level" value={level} onChangeText={setLevel} placeholder="beginner / intermediate / advanced" />
            </Section>
            <Button title={loading ? "Starting…" : "Start Lesson"} onPress={startLesson} disabled={loading} />
            {loading ? (
              <View style={{ marginTop: 12 }}>
                <ActivityIndicator />
              </View>
            ) : null}
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
          <ScrollView ref={scrollRef} onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}>
            {messages.map((m, idx) => (
              m.kind === "bubble" ? (
                <Bubble key={idx} role={m.role} text={m.text} />
              ) : (
                <ExerciseCard key={idx} ex={m.ex} onAnswer={onAnswerExercise} />
              )
            ))}
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
