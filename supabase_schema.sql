-- ============================================================================
-- Supabase Schema for SMART-BP Blood Pressure Tracker
-- Run this in your Supabase Project SQL Editor to set up cloud persistence.
-- ============================================================================

-- 1. Create table for blood pressure readings
CREATE TABLE IF NOT EXISTS public.bp_readings (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id TEXT DEFAULT 'personal_owner' NOT NULL,
    timestamp TEXT NOT NULL,
    sys INTEGER NOT NULL CHECK (sys >= 40 AND sys <= 260),
    dia INTEGER NOT NULL CHECK (dia >= 30 AND dia <= 200),
    pulse INTEGER CHECK (pulse >= 25 AND pulse <= 240),
    category TEXT,
    model_variant TEXT DEFAULT 'SMART-BP+',
    confidence NUMERIC,
    original_sys INTEGER,
    original_dia INTEGER,
    original_pulse INTEGER,
    was_corrected BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- 2. Create indices for fast chronological history querying and user separation
CREATE INDEX IF NOT EXISTS idx_bp_readings_user_time ON public.bp_readings (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bp_readings_timestamp ON public.bp_readings (timestamp DESC);

-- 3. Enable Row Level Security (RLS) for data protection
ALTER TABLE public.bp_readings ENABLE ROW LEVEL SECURITY;

-- 4. Policy: Allow service role (backend API) full access
-- NOTE: The backend application strictly enforces user scoping at the query layer (.eq("user_id", ...))
CREATE POLICY "Allow service role full access"
    ON public.bp_readings
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
