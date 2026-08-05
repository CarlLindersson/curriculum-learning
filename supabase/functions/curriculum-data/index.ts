import { createClient } from "npm:@supabase/supabase-js@2";

const MAX_BODY_BYTES = 250_000;
const MAX_BATCH_SIZE = 20;
const PARTICIPANT_RE = /^[A-Za-z0-9_.-]{1,40}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const ALIAS_RE = /^[a-z]+-[a-z]+$/;
// `x` and `hexagon` remain valid for older page instances that were already loaded.
const SHAPES = new Set([
  "circle",
  "square",
  "triangle",
  "star",
  "pentagon",
  "plus",
  "x",
  "hexagon",
]);
const SIZES = new Set(["small", "large"]);
const CURRICULA = new Set([
  "interleaved",
  "blocked",
  "progressively_interleaved",
  "progressively_blocked",
]);

const supabaseUrl = Deno.env.get("SUPABASE_URL");
const secretKeys = JSON.parse(Deno.env.get("SUPABASE_SECRET_KEYS") ?? "{}");
const serviceRoleKey = secretKeys.default ?? Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
if (!supabaseUrl || !serviceRoleKey) {
  throw new Error("Supabase runtime secrets are unavailable");
}
const db = createClient(supabaseUrl, serviceRoleKey, {
  auth: { persistSession: false, autoRefreshToken: false },
});

function corsHeaders(request: Request): Record<string, string> {
  const origin = request.headers.get("origin") ?? "";
  const configured = (Deno.env.get("ALLOWED_ORIGINS") ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const allowOrigin = configured.includes(origin) ? origin : "";
  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Vary": "Origin",
  };
}

function json(request: Request, status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(request), "Content-Type": "application/json" },
  });
}

function fail(message: string): never {
  throw new Error(message);
}

function requiredString(value: unknown, name: string, max = 80): string {
  if (typeof value !== "string" || value.length === 0 || value.length > max) {
    fail(`${name} must be a non-empty string of at most ${max} characters`);
  }
  return value as string;
}

function requiredInteger(value: unknown, name: string, min = 0, max = 1_000_000): number {
  if (!Number.isInteger(value) || (value as number) < min || (value as number) > max) {
    fail(`${name} must be an integer from ${min} to ${max}`);
  }
  return value as number;
}

function symbolName(size: string, shape: string): string {
  return `${size}-${shape}`;
}

type SessionRow = {
  participant_id: string;
  curriculum: string;
  session_number: number;
  completed_at: string | null;
};

type AliasOptionRow = {
  alias: string;
  is_current: boolean;
};

function aliasOffer(raw: unknown) {
  if (!Array.isArray(raw)) fail("Invalid leaderboard-name response");
  const rows = raw as AliasOptionRow[];
  for (const row of rows) {
    if (!row || typeof row.alias !== "string" || !ALIAS_RE.test(row.alias) ||
        typeof row.is_current !== "boolean") {
      fail("Invalid leaderboard-name response");
    }
  }
  const current = rows.find((row) => row.is_current);
  if (current) {
    return { leaderboard_name: current.alias, alias_options: [] as string[] };
  }
  const options = [...new Set(rows.map((row) => row.alias))].slice(0, 3);
  if (options.length === 0) fail("No leaderboard names remain");
  return { leaderboard_name: null, alias_options: options };
}

function validateTrial(raw: unknown, sessionId: string, session: SessionRow) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) fail("Each trial must be an object");
  const trial = raw as Record<string, unknown>;
  const phase = requiredString(trial.phase, "phase");
  if (!new Set(["training", "test"]).has(phase)) fail("Invalid phase");
  const operation = requiredString(trial.operation, "operation");
  if (!new Set(["size", "shape"]).has(operation)) fail("Invalid operation");

  const startShape = requiredString(trial.start_shape, "start_shape");
  const correctShape = requiredString(trial.correct_shape, "correct_shape");
  const topShape = requiredString(trial.top_shape, "top_shape");
  const bottomShape = requiredString(trial.bottom_shape, "bottom_shape");
  for (const shape of [startShape, correctShape, topShape, bottomShape]) {
    if (!SHAPES.has(shape)) fail("Invalid shape");
  }
  const startSize = requiredString(trial.start_size, "start_size");
  const correctSize = requiredString(trial.correct_size, "correct_size");
  const topSize = requiredString(trial.top_size, "top_size");
  const bottomSize = requiredString(trial.bottom_size, "bottom_size");
  for (const size of [startSize, correctSize, topSize, bottomSize]) {
    if (!SIZES.has(size)) fail("Invalid size");
  }

  const response = requiredString(trial.response, "response");
  if (!new Set(["top", "bottom", "timeout"]).has(response)) fail("Invalid response");
  if (typeof trial.timeout !== "boolean") fail("timeout must be boolean");
  const timeout = trial.timeout;
  if (timeout !== (response === "timeout")) fail("timeout and response disagree");
  const responseType = requiredString(trial.response_type, "response_type");
  if (!new Set(["mouse", "arrow", "none", "unknown"]).has(responseType)) {
    fail("Invalid response_type");
  }

  let responseTime: number | null = null;
  if (trial.response_time_ms !== "" && trial.response_time_ms != null) {
    responseTime = requiredInteger(trial.response_time_ms, "response_time_ms", 0, 60_000);
  }
  if (timeout && responseTime !== null) fail("Timed-out trials cannot have a response latency");

  const startSymbol = symbolName(startSize, startShape);
  const topSymbol = symbolName(topSize, topShape);
  const bottomSymbol = symbolName(bottomSize, bottomShape);
  const correctSymbol = symbolName(correctSize, correctShape);
  if (trial.start_symbol !== startSymbol) fail("start_symbol disagrees with its dimensions");
  if (!Array.isArray(trial.option_symbols) || trial.option_symbols.length !== 2 ||
      trial.option_symbols[0] !== topSymbol || trial.option_symbols[1] !== bottomSymbol) {
    fail("option_symbols disagree with the option dimensions");
  }
  const expectedChoice = response === "top" ? topSymbol : response === "bottom" ? bottomSymbol : null;
  const suppliedChoice = trial.choice === "" || trial.choice == null ? null : trial.choice;
  if (suppliedChoice !== expectedChoice) fail("choice disagrees with response");
  const expectedCorrect = expectedChoice !== null && expectedChoice === correctSymbol;
  if (typeof trial.is_correct !== "boolean" || trial.is_correct !== expectedCorrect) {
    fail("is_correct disagrees with the selected and correct symbols");
  }

  const timestamp = requiredString(trial.timestamp_utc, "timestamp_utc");
  const timestampMs = Date.parse(timestamp);
  if (!Number.isFinite(timestampMs)) fail("timestamp_utc must be an ISO timestamp");

  return {
    session_id: sessionId,
    participant_id: session.participant_id,
    curriculum: session.curriculum,
    phase,
    trial_number: requiredInteger(trial.trial_number, "trial_number", 1),
    operation,
    start_shape: startShape,
    start_size: startSize,
    correct_shape: correctShape,
    correct_size: correctSize,
    top_shape: topShape,
    top_size: topSize,
    bottom_shape: bottomShape,
    bottom_size: bottomSize,
    response,
    is_correct: expectedCorrect,
    response_time_ms: responseTime,
    score: requiredInteger(trial.score, "score"),
    timestamp_utc: new Date(timestampMs).toISOString(),
    session_number: session.session_number,
    start_symbol: startSymbol,
    option_symbols: [topSymbol, bottomSymbol],
    choice: expectedChoice,
    timeout,
    curriculum_phase: requiredString(trial.curriculum_phase, "curriculum_phase"),
    response_type: responseType,
  };
}

Deno.serve(async (request: Request) => {
  if (request.method === "OPTIONS") return new Response("ok", { headers: corsHeaders(request) });
  if (request.method !== "POST") return json(request, 405, { ok: false, error: "POST required" });
  if (corsHeaders(request)["Access-Control-Allow-Origin"] === "") {
    return json(request, 403, { ok: false, error: "Origin is not allowed" });
  }
  const contentLength = Number(request.headers.get("content-length") ?? "0");
  if (contentLength > MAX_BODY_BYTES) return json(request, 413, { ok: false, error: "Request too large" });

  try {
    const text = await request.text();
    if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) fail("Request too large");
    const body = JSON.parse(text) as Record<string, unknown>;
    const action = requiredString(body.action, "action");

    if (action === "identify_participant") {
      const participantId = requiredString(body.participant_id, "participant_id", 40);
      if (!PARTICIPANT_RE.test(participantId)) fail("Invalid participant_id");
      const { data: aliases, error: aliasError } = await db.rpc(
        "get_curriculum_alias_options",
        { p_participant_id: participantId },
      );
      if (aliasError) throw aliasError;
      return json(request, 200, { ok: true, ...aliasOffer(aliases) });
    }

    const sessionId = requiredString(body.session_id, "session_id");
    if (!UUID_RE.test(sessionId)) fail("Invalid session_id");

    if (action === "start_session") {
      const participantId = requiredString(body.participant_id, "participant_id", 40);
      const curriculum = requiredString(body.curriculum, "curriculum");
      if (!PARTICIPANT_RE.test(participantId)) fail("Invalid participant_id");
      if (!CURRICULA.has(curriculum)) fail("Invalid curriculum");
      const { data, error } = await db.rpc("start_curriculum_session", {
        p_session_id: sessionId,
        p_participant_id: participantId,
        p_curriculum: curriculum,
      });
      if (error) throw error;
      const { data: aliases, error: aliasError } = await db.rpc(
        "get_curriculum_alias_options",
        { p_participant_id: participantId },
      );
      if (aliasError) throw aliasError;
      return json(request, 200, {
        ok: true,
        session_number: data,
        ...aliasOffer(aliases),
      });
    }

    const { data: session, error: sessionError } = await db
      .from("curriculum_sessions")
      .select("participant_id,curriculum,session_number,completed_at")
      .eq("id", sessionId)
      .single();
    if (sessionError || !session) fail("Unknown session");

    if (action === "claim_alias") {
      const alias = requiredString(body.alias, "alias");
      if (!ALIAS_RE.test(alias)) fail("Invalid alias");
      const { data, error } = await db.rpc("claim_curriculum_alias", {
        p_participant_id: session.participant_id,
        p_alias: alias,
      });
      if (error) throw error;
      const result = Array.isArray(data) ? data[0] : data;
      if (result?.claimed === true) {
        const leaderboardName = requiredString(
          result.leaderboard_name,
          "leaderboard_name",
        );
        if (!ALIAS_RE.test(leaderboardName)) fail("Invalid leaderboard name");
        return json(request, 200, {
          ok: true,
          claimed: true,
          leaderboard_name: leaderboardName,
          alias_options: [],
        });
      }

      const { data: aliases, error: aliasError } = await db.rpc(
        "get_curriculum_alias_options",
        { p_participant_id: session.participant_id },
      );
      if (aliasError) throw aliasError;
      return json(request, 200, {
        ok: true,
        claimed: false,
        ...aliasOffer(aliases),
      });
    }

    if (action === "save_trials") {
      if (session.completed_at) fail("Session is already complete");
      if (!Array.isArray(body.trials) || body.trials.length === 0 || body.trials.length > MAX_BATCH_SIZE) {
        fail(`trials must contain 1 to ${MAX_BATCH_SIZE} rows`);
      }
      const rows = body.trials.map((trial) => validateTrial(trial, sessionId, session as SessionRow));
      const { error } = await db.from("curriculum_trials").upsert(rows, {
        onConflict: "session_id,trial_number",
      });
      if (error) throw error;
      return json(request, 200, { ok: true, saved: rows.length });
    }

    if (action === "complete_session") {
      const { data, error } = await db.rpc("complete_curriculum_session", {
        p_session_id: sessionId,
      });
      if (error) throw error;
      const result = Array.isArray(data) ? data[0] : data;
      return json(request, 200, {
        ok: true,
        ranking: Number(result.ranking),
        score: Number(result.final_score),
      });
    }

    return json(request, 400, { ok: false, error: "Unknown action" });
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : "Invalid request";
    const status = message === "Request too large" ? 413 : 400;
    return json(request, status, { ok: false, error: message });
  }
});
