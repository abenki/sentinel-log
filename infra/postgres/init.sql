CREATE TABLE IF NOT EXISTS anomalies (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    log_type    VARCHAR(16) NOT NULL,
    source      VARCHAR(255) NOT NULL,
    score       FLOAT NOT NULL,
    threshold   FLOAT NOT NULL,
    payload     JSONB NOT NULL,
    model_name  VARCHAR(128),
    model_version VARCHAR(32)
);

CREATE INDEX IF NOT EXISTS idx_anomalies_detected_at ON anomalies (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_log_type ON anomalies (log_type);
