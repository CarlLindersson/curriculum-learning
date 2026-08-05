-- Persistent pseudonymous identities and unique leaderboard names.
-- The browser stores only an opaque UUID. Names are offered and claimed through
-- SECURITY DEFINER functions called by the Edge Function, never by public table access.

create table if not exists public.curriculum_participants (
    participant_id text primary key
        check (participant_id ~ '^[A-Za-z0-9_.-]{1,40}$'),
    created_at timestamptz not null default now()
);

insert into public.curriculum_participants (participant_id)
select distinct participant_id
  from public.curriculum_sessions
on conflict (participant_id) do nothing;

create table if not exists public.curriculum_aliases (
    alias text primary key check (alias ~ '^[a-z]+-[a-z]+$'),
    claimed_by text unique
        references public.curriculum_participants(participant_id) on delete set null,
    claimed_at timestamptz,
    check (
        (claimed_by is null and claimed_at is null)
        or (claimed_by is not null and claimed_at is not null)
    )
);

with adjectives(adjective) as (
    select unnest(array[
        'adventurous', 'amazing', 'amusing', 'bold', 'breezy',
        'bright', 'bubbly', 'calm', 'charming', 'cheerful',
        'clever', 'cosmic', 'cozy', 'curious', 'dapper',
        'daring', 'delightful', 'dreamy', 'eager', 'elegant',
        'fabulous', 'fluffy', 'friendly', 'gentle', 'gleaming',
        'happy', 'helpful', 'jolly', 'kind', 'lively',
        'lucky', 'merry', 'mighty', 'nimble', 'playful',
        'polite', 'quirky', 'radiant', 'silly', 'sparkly',
        'speedy', 'sunny', 'superb', 'swift', 'thoughtful',
        'tiny', 'vibrant', 'witty', 'wonderful', 'zany'
    ]::text[])
), animals(animal) as (
    select unnest(array[
        'alpaca', 'badger', 'beaver', 'bumblebee', 'butterfly',
        'capybara', 'cat', 'cheetah', 'chipmunk', 'dolphin',
        'donkey', 'duck', 'elephant', 'falcon', 'ferret',
        'fox', 'frog', 'gecko', 'giraffe', 'goat',
        'hedgehog', 'heron', 'hippo', 'koala', 'lemur',
        'lion', 'llama', 'lobster', 'manatee', 'meerkat',
        'mole', 'mongoose', 'mouse', 'narwhal', 'otter',
        'owl', 'panda', 'parrot', 'penguin', 'porpoise',
        'rabbit', 'raccoon', 'seal', 'sloth', 'sparrow',
        'squirrel', 'tiger', 'turtle', 'walrus', 'wombat'
    ]::text[])
)
insert into public.curriculum_aliases (alias)
select adjectives.adjective || '-' || animals.animal
  from adjectives cross join animals
on conflict (alias) do nothing;

alter table public.curriculum_sessions
    add column if not exists leaderboard_name text;

do $$
begin
    if not exists (
        select 1
          from pg_constraint
         where conname = 'curriculum_sessions_leaderboard_name_fkey'
           and conrelid = 'public.curriculum_sessions'::regclass
    ) then
        alter table public.curriculum_sessions
            add constraint curriculum_sessions_leaderboard_name_fkey
            foreign key (leaderboard_name)
            references public.curriculum_aliases(alias);
    end if;
end;
$$;

alter table public.curriculum_participants enable row level security;
alter table public.curriculum_aliases enable row level security;

revoke all on public.curriculum_participants from anon, authenticated;
revoke all on public.curriculum_aliases from anon, authenticated;

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

    insert into public.curriculum_participants (participant_id)
    values (p_participant_id)
    on conflict (participant_id) do nothing;

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
        id, participant_id, session_number, curriculum, leaderboard_name
    )
    select p_session_id,
           p_participant_id,
           v_session_number,
           p_curriculum,
           aliases.alias
      from (select 1) as singleton
      left join public.curriculum_aliases as aliases
        on aliases.claimed_by = p_participant_id;

    return v_session_number;
end;
$$;

create or replace function public.get_curriculum_alias_options(
    p_participant_id text
) returns table (alias text, is_current boolean)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_existing_alias text;
begin
    select names.alias
      into v_existing_alias
      from public.curriculum_aliases as names
     where names.claimed_by = p_participant_id;

    if found then
        return query select v_existing_alias, true;
        return;
    end if;

    return query
        select names.alias, false
          from public.curriculum_aliases as names
         where names.claimed_by is null
         order by random()
         limit 3;
end;
$$;

create or replace function public.claim_curriculum_alias(
    p_participant_id text,
    p_alias text
) returns table (claimed boolean, leaderboard_name text)
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_alias text;
begin
    perform pg_advisory_xact_lock(hashtextextended(p_participant_id, 0));

    select names.alias
      into v_alias
      from public.curriculum_aliases as names
     where names.claimed_by = p_participant_id;

    if found then
        return query select true, v_alias;
        return;
    end if;

    update public.curriculum_aliases as names
       set claimed_by = p_participant_id,
           claimed_at = now()
     where names.alias = p_alias
       and names.claimed_by is null
    returning names.alias into v_alias;

    if v_alias is null then
        return query select false, null::text;
        return;
    end if;

    update public.curriculum_sessions
       set leaderboard_name = v_alias
     where participant_id = p_participant_id;

    return query select true, v_alias;
end;
$$;

-- PostgREST still requires table privileges for the Edge Function's direct reads
-- and writes even though service_role bypasses row-level security.
grant usage on schema public to service_role;
grant select, insert, update on public.curriculum_sessions to service_role;
grant select, insert, update on public.curriculum_trials to service_role;

revoke all on function public.start_curriculum_session(uuid, text, text)
    from public, anon, authenticated;
revoke all on function public.get_curriculum_alias_options(text)
    from public, anon, authenticated;
revoke all on function public.claim_curriculum_alias(text, text)
    from public, anon, authenticated;

grant execute on function public.start_curriculum_session(uuid, text, text)
    to service_role;
grant execute on function public.get_curriculum_alias_options(text)
    to service_role;
grant execute on function public.claim_curriculum_alias(text, text)
    to service_role;

notify pgrst, 'reload schema';
