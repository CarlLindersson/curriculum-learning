-- Allocate the two online curricula on the server while preserving all existing
-- session assignments. Existing session counts seed the catch-up behavior.

create or replace function public.start_balanced_curriculum_session(
    p_session_id uuid,
    p_participant_id text
) returns table (session_number integer, curriculum text)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_session_number integer;
    v_existing_participant text;
    v_existing_curriculum text;
    v_curriculum text;
    v_blocked_count bigint;
    v_progressive_count bigint;
begin
    -- Serialize allocation and insertion so simultaneous arrivals see the latest
    -- counts and cannot both claim the same balancing position.
    perform pg_advisory_xact_lock(
        hashtextextended('curriculum-balanced-allocation-v1', 0)
    );

    select sessions.session_number,
           sessions.participant_id,
           sessions.curriculum
      into v_session_number,
           v_existing_participant,
           v_existing_curriculum
      from public.curriculum_sessions as sessions
     where sessions.id = p_session_id;

    if found then
        if v_existing_participant <> p_participant_id then
            raise exception 'Session identifier is already in use';
        end if;
        return query select v_session_number, v_existing_curriculum;
        return;
    end if;

    insert into public.curriculum_participants (participant_id)
    values (p_participant_id)
    on conflict (participant_id) do nothing;

    select count(*) filter (
               where sessions.curriculum = 'blocked'
           ),
           count(*) filter (
               where sessions.curriculum = 'progressively_interleaved'
           )
      into v_blocked_count, v_progressive_count
      from public.curriculum_sessions as sessions;

    if v_blocked_count < v_progressive_count then
        v_curriculum := 'blocked';
    elsif v_progressive_count < v_blocked_count then
        v_curriculum := 'progressively_interleaved';
    elsif random() < 0.5 then
        v_curriculum := 'blocked';
    else
        v_curriculum := 'progressively_interleaved';
    end if;

    select coalesce(max(sessions.session_number), 0) + 1
      into v_session_number
      from public.curriculum_sessions as sessions
     where sessions.participant_id = p_participant_id;

    insert into public.curriculum_sessions (
        id, participant_id, session_number, curriculum, leaderboard_name
    )
    select p_session_id,
           p_participant_id,
           v_session_number,
           v_curriculum,
           aliases.alias
      from (select 1) as singleton
      left join public.curriculum_aliases as aliases
        on aliases.claimed_by = p_participant_id;

    return query select v_session_number, v_curriculum;
end;
$$;

revoke all on function public.start_balanced_curriculum_session(uuid, text)
    from public, anon, authenticated;
grant execute on function public.start_balanced_curriculum_session(uuid, text)
    to service_role;
