-- Initialize PostgreSQL database for RCA system

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Cases table
CREATE TABLE IF NOT EXISTS cases (
    case_id VARCHAR(50) PRIMARY KEY,
    failure_type VARCHAR(100) NOT NULL,
    failure_description TEXT NOT NULL,
    failure_datetime TIMESTAMP NOT NULL,
    part_number VARCHAR(100) NOT NULL,
    serial_number VARCHAR(100),
    lot_number VARCHAR(100),
    station_id VARCHAR(50),
    line_id VARCHAR(50),
    shift VARCHAR(20),
    operator_id VARCHAR(50),
    test_name VARCHAR(100),
    test_value FLOAT,
    spec_lower FLOAT,
    spec_upper FLOAT,
    component_lots JSONB,
    status VARCHAR(50) DEFAULT 'intake',
    report_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reports table
CREATE TABLE IF NOT EXISTS reports (
    report_id VARCHAR(50) PRIMARY KEY,
    case_id VARCHAR(50) REFERENCES cases(case_id),
    title TEXT NOT NULL,
    report_data JSONB NOT NULL,
    overall_confidence FLOAT,
    top_hypothesis_id VARCHAR(50),
    is_draft BOOLEAN DEFAULT TRUE,
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id VARCHAR(50) PRIMARY KEY,
    report_id VARCHAR(50) REFERENCES reports(report_id),
    case_id VARCHAR(50) REFERENCES cases(case_id),
    engineer_id VARCHAR(100) NOT NULL,
    report_useful BOOLEAN NOT NULL,
    time_saved_minutes INTEGER,
    actual_root_cause TEXT,
    root_cause_was_in_top_3 BOOLEAN DEFAULT FALSE,
    correct_hypothesis_id VARCHAR(50),
    hypothesis_feedback JSONB,
    stats_feedback JSONB,
    citation_feedback JSONB,
    what_worked_well TEXT,
    what_to_improve TEXT,
    additional_notes TEXT,
    should_update_recipe BOOLEAN DEFAULT FALSE,
    suggested_recipe_changes TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks table for RAG
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id VARCHAR(100) PRIMARY KEY,
    document_id VARCHAR(100) NOT NULL,
    document_name VARCHAR(255) NOT NULL,
    document_type VARCHAR(50) DEFAULT 'product_guide',
    content TEXT NOT NULL,
    section_path TEXT,
    page_number INTEGER,
    start_line INTEGER,
    end_line INTEGER,
    revision VARCHAR(50),
    effective_date DATE,
    embedding vector(1536),  -- Adjust dimension based on your embedding model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id VARCHAR(50) PRIMARY KEY,
    evaluation_timestamp TIMESTAMP NOT NULL,
    total_metrics INTEGER,
    passed_metrics INTEGER,
    failed_metrics INTEGER,
    approved BOOLEAN,
    blocking_failures JSONB,
    case_count INTEGER,
    feedback_count INTEGER,
    metrics_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow checkpoints table
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    checkpoint_id VARCHAR(50) PRIMARY KEY,
    workflow_id VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    checkpoint_timestamp TIMESTAMP NOT NULL,
    context JSONB,
    completed_steps JSONB,
    outputs JSONB,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    log_id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    case_id VARCHAR(50),
    agent_name VARCHAR(100),
    action VARCHAR(255),
    details JSONB,
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_cases_failure_type ON cases(failure_type);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_case_id ON reports(case_id);
CREATE INDEX IF NOT EXISTS idx_feedback_report_id ON feedback(report_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_audit_case_id ON audit_log(case_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
