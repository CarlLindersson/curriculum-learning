-- Online storage for the curriculum-learning pilot.
-- Browser clients never access these tables directly; only the Edge Function's
-- service-role client may read or write them.

create table if not exists public.curriculum_sessions (
    id uuid primary key,
    participant_id text not null
        check (participant_id ~ '^[A-Za-z0-9_.-]{1,40}$'),
    session_number integer not null check (session_number > 0),
    curriculum text not null check (
        curriculum in (
            'interleaved',
            'blocked',
            'progressively_interleaved',
            'progressively_blocked'
        )
    ),
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    score integer check (score >= 0),
    total_trials integer check (total_trials >= 0),
    total_accuracy double precision check (total_accuracy between 0 and 1),
    training_trials integer check (training_trials >= 0),
    training_accuracy double precision check (training_accuracy between 0 and 1),
    test_trials integer check (test_trials >= 0),
    test_accuracy double precision check (test_accuracy between 0 and 1),
    unique (participant_id, session_number)
);

create table if not exists public.curriculum_trials (
    session_id uuid not null references public.curriculum_sessions(id) on delete cascade,
    participant_id text not null,
    curriculum text not null,
    phase text not null check (phase in ('training', 'test')),
    trial_number integer not null check (trial_number > 0),
    operation text not null check (operation in ('size', 'shape')),
    start_shape text not null,
    start_size text not null check (start_size in ('small', 'large')),
    correct_shape text not null,
    correct_size text not null check (correct_size in ('small', 'large')),
    top_shape text not null,
    top_size text not null check (top_size in ('small', 'large')),
    bottom_shape text not null,
    bottom_size text not null check (bottom_size in ('small', 'large')),
    response text not null check (response in ('top', 'bottom', 'timeout')),
    is_correct boolean not null,
    response_time_ms integer check (response_time_ms between 0 and 60000),
    score integer not null check (score >= 0),
    timestamp_utc timestamptz not null,
    session_number integer not null check (session_number > 0),
    start_symbol text not null,
    option_symbols jsonb not null check (
        jsonb_typeof(option_symbols) = 'array'
        and jsonb_array_length(option_symbols) = 2
    ),
    choice text,
    timeout boolean not null,
    curriculum_phase text not null,
    response_type text not null check (
        response_type in ('mouse', 'arrow', 'none', 'unknown')
    ),
    primary key (session_id, trial_number),
    check ((timeout and response = 'timeout') or (not timeout and response <> 'timeout')),
    check ((timeout and response_time_ms is null) or not timeout)
);

create index if not exists curriculum_sessions_participant_idx
    on public.curriculum_sessions (participant_id, session_number);
create index if not exists curriculum_sessions_ranking_idx
    on public.curriculum_sessions (score desc) where completed_at is not null;
create index if not exists curriculum_trials_session_idx
    on public.curriculum_trials (session_id, trial_number);

alter table public.curriculum_sessions enable row level security;
alter table public.curriculum_trials enable row level security;

revoke all on public.curriculum_sessions from anon, authenticated;
revoke all on public.curriculum_trials from anon, authenticated;

create or replace function public.start_curriculum_session(
    p_session_id uuid,
    p_participant_id text,
    p_curriculum text
) returns integer
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_session_number integer;
    v_existing_participant text;
    v_existing_curriculum text;
begin
    perform pg_advisory_xact_lock(hashtextextended(p_participant_id, 0));

    select session_number, participant_id, curriculum
      into v_session_number, v_existing_participant, v_existing_curriculum
      from public.curriculum_sessions
     where id = p_session_id;

    if found then
        if v_existing_participant <> p_participant_id
           or v_existing_curriculum <> p_curriculum then
            raise exception 'Session identifier is already in use';
        end if;
        return v_session_number;
    end if;

    select coalesce(max(session_number), 0) + 1
      into v_session_number
      from public.curriculum_sessions
     where participant_id = p_participant_id;

    insert into public.curriculum_sessions (
        id, participant_id, session_number, curriculum
    ) values (
        p_session_id, p_participant_id, v_session_number, p_curriculum
    );

    return v_session_number;
end;
$$;

create or replace function public.complete_curriculum_session(p_session_id uuid)
returns table (ranking bigint, final_score integer)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_score integer;
begin
    select coalesce(count(*) filter (where is_correct), 0)::integer
      into v_score
      from public.curriculum_trials
     where session_id = p_session_id;

    update public.curriculum_sessions as sessions
       set completed_at = coalesce(sessions.completed_at, now()),
           score = v_score,
           total_trials = stats.total_trials,
           total_accuracy = stats.total_accuracy,
           training_trials = stats.training_trials,
           training_accuracy = stats.training_accuracy,
           test_trials = stats.test_trials,
           test_accuracy = stats.test_accuracy
      from (
          select count(*)::integer as total_trials,
                 coalesce(avg(is_correct::integer), 0)::double precision
                     as total_accuracy,
                 count(*) filter (where phase = 'training')::integer
                     as training_trials,
                 coalesce(avg(is_correct::integer) filter (where phase = 'training'), 0)
                     ::double precision as training_accuracy,
                 count(*) filter (where phase = 'test')::integer as test_trials,
                 coalesce(avg(is_correct::integer) filter (where phase = 'test'), 0)
                     ::double precision as test_accuracy
            from public.curriculum_trials
           where session_id = p_session_id
      ) as stats
     where sessions.id = p_session_id;

    if not found then
        raise exception 'Unknown session';
    end if;

    return query
        select (1 + count(*))::bigint, v_score
          from public.curriculum_sessions
         where completed_at is not null and score > v_score;
end;
$$;

revoke all on function public.start_curriculum_session(uuid, text, text)
    from public, anon, authenticated;
revoke all on function public.complete_curriculum_session(uuid)
    from public, anon, authenticated;
grant execute on function public.start_curriculum_session(uuid, text, text)
    to service_role;
grant execute on function public.complete_curriculum_session(uuid)
    to service_role;
