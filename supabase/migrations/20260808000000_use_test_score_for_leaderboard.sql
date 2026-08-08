-- Rank completed sessions by correct test trials only. Training performance is
-- retained in the trial and summary data but no longer contributes to the
-- leaderboard score.

update public.curriculum_sessions as sessions
   set score = coalesce(
       (
           select count(*)::integer
             from public.curriculum_trials as trials
            where trials.session_id = sessions.id
              and trials.phase = 'test'
              and trials.is_correct
       ),
       0
   )
 where sessions.completed_at is not null;

create or replace function public.complete_curriculum_session(p_session_id uuid)
returns table (ranking bigint, final_score integer)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_score integer;
begin
    select coalesce(
               count(*) filter (where is_correct and phase = 'test'),
               0
           )::integer
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
                 coalesce(
                     avg(is_correct::integer) filter (where phase = 'training'),
                     0
                 )::double precision as training_accuracy,
                 count(*) filter (where phase = 'test')::integer as test_trials,
                 coalesce(
                     avg(is_correct::integer) filter (where phase = 'test'),
                     0
                 )::double precision as test_accuracy
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

revoke all on function public.complete_curriculum_session(uuid)
    from public, anon, authenticated;
grant execute on function public.complete_curriculum_session(uuid)
    to service_role;
