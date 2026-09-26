-- ============================================================================
-- Supabase Schema for SMART-BP Blood Pressure Tracker
-- Run this in your Supabase Project SQL Editor to set up cloud persistence.
-- ============================================================================

-- 1. Create table for blood pressure readings
CREATE TABLE IF NOT EXISTS public.bp_readings (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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

-- 2. Create index for fast chronological history querying
CREATE INDEX IF NOT EXISTS idx_bp_readings_timestamp ON public.bp_readings (timestamp DESC);

-- 3. Enable Row Level Security (RLS) for data protection
ALTER TABLE public.bp_readings ENABLE ROW LEVEL SECURITY;

-- 4. Policy: Allow service role (backend API) full access
CREATE POLICY "Allow service role full access"
    ON public.bp_readings
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- 5. Optional Policy: If using authenticated users in the future
-- ALTER TABLE public.bp_readings ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id);
-- CREATE POLICY "Users can only view their own readings"
--     ON public.bp_readings FOR SELECT
--     USING (auth.uid() = user_id);
