-- ============================================================================
-- Vehicle Analytics Database Schema
-- PostgreSQL 12+
-- ============================================================================

-- Create database (run as superuser)
-- CREATE DATABASE vehicle_analytics WITH ENCODING 'UTF8';

-- Connect to the database
\c vehicle_analytics;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search on license plates

-- ============================================================================
-- TABLE: cameras
-- Stores metadata about deployed CCTV cameras
-- ============================================================================

CREATE TABLE IF NOT EXISTS cameras (
    camera_id SERIAL PRIMARY KEY,
    camera_name VARCHAR(100) NOT NULL UNIQUE,
    location_name VARCHAR(200) NOT NULL,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    stream_url VARCHAR(500),
    installation_date DATE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add sample camera
INSERT INTO cameras (camera_name, location_name, latitude, longitude, stream_url, installation_date)
VALUES 
    ('ROUNDABOUT_CAM_01', 'Main Street Roundabout - North Entrance', 18.5204, 73.8567, 'rtsp://admin:password@192.168.1.100:554/stream1', '2025-01-01'),
    ('ROUNDABOUT_CAM_02', 'Main Street Roundabout - South Exit', 18.5200, 73.8565, 'rtsp://admin:password@192.168.1.101:554/stream1', '2025-01-01')
ON CONFLICT (camera_name) DO NOTHING;

-- ============================================================================
-- TABLE: vehicle_sightings
-- Core table storing all vehicle detection events
-- ============================================================================

CREATE TABLE IF NOT EXISTS vehicle_sightings (
    sighting_id BIGSERIAL PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    license_plate VARCHAR(15) NOT NULL,
    vehicle_color VARCHAR(20),
    direction VARCHAR(20) CHECK (direction IN ('towards_camera', 'away_from_camera', 'unknown')),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    vehicle_type VARCHAR(20),  -- car, truck, bus, motorcycle, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES for Performance Optimization
-- ============================================================================

-- Primary query patterns: time-range searches and license plate lookups

-- Index on timestamp for time-based queries (most common)
CREATE INDEX IF NOT EXISTS idx_sightings_timestamp 
ON vehicle_sightings(timestamp DESC);

-- Index on license plate for vehicle tracking queries
CREATE INDEX IF NOT EXISTS idx_sightings_plate 
ON vehicle_sightings(license_plate);

-- Composite index for camera + timestamp (for per-camera analytics)
CREATE INDEX IF NOT EXISTS idx_sightings_camera_timestamp 
ON vehicle_sightings(camera_id, timestamp DESC);

-- Composite index for license plate + timestamp (for vehicle journey reconstruction)
CREATE INDEX IF NOT EXISTS idx_sightings_plate_timestamp 
ON vehicle_sightings(license_plate, timestamp DESC);

-- GIN index for fuzzy text search on license plates (handles OCR errors)
CREATE INDEX IF NOT EXISTS idx_sightings_plate_trgm 
ON vehicle_sightings USING gin(license_plate gin_trgm_ops);

-- Index on direction for directional traffic analysis
CREATE INDEX IF NOT EXISTS idx_sightings_direction 
ON vehicle_sightings(direction) WHERE direction IS NOT NULL;

-- Partial index for high-confidence detections only
CREATE INDEX IF NOT EXISTS idx_sightings_high_confidence 
ON vehicle_sightings(timestamp DESC) 
WHERE confidence > 0.8;

-- ============================================================================
-- TABLE: vehicle_journeys (Optional - for multi-camera tracking)
-- Aggregates multiple sightings into journeys across cameras
-- ============================================================================

CREATE TABLE IF NOT EXISTS vehicle_journeys (
    journey_id BIGSERIAL PRIMARY KEY,
    license_plate VARCHAR(15) NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL,
    last_seen TIMESTAMPTZ NOT NULL,
    camera_sequence INTEGER[],  -- Array of camera IDs in order
    total_sightings INTEGER DEFAULT 0,
    journey_duration_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_journeys_plate ON vehicle_journeys(license_plate);
CREATE INDEX IF NOT EXISTS idx_journeys_first_seen ON vehicle_journeys(first_seen DESC);

-- ============================================================================
-- TABLE: analytics_summary (Materialized View for Fast Dashboards)
-- Pre-computed hourly statistics
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS analytics_summary AS
SELECT 
    camera_id,
    DATE_TRUNC('hour', timestamp) AS hour,
    COUNT(*) AS total_vehicles,
    COUNT(DISTINCT license_plate) AS unique_vehicles,
    COUNT(*) FILTER (WHERE direction = 'towards_camera') AS vehicles_towards,
    COUNT(*) FILTER (WHERE direction = 'away_from_camera') AS vehicles_away,
    AVG(confidence) AS avg_confidence,
    COUNT(*) FILTER (WHERE vehicle_type = 'car') AS cars,
    COUNT(*) FILTER (WHERE vehicle_type = 'truck') AS trucks,
    COUNT(*) FILTER (WHERE vehicle_type = 'motorcycle') AS motorcycles,
    COUNT(*) FILTER (WHERE vehicle_type = 'bus') AS buses
FROM vehicle_sightings
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY camera_id, DATE_TRUNC('hour', timestamp);

-- Index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_summary_unique 
ON analytics_summary(camera_id, hour);

-- Refresh function (call this periodically via cron)
-- Example: SELECT refresh_analytics_summary();
CREATE OR REPLACE FUNCTION refresh_analytics_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics_summary;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TABLE: alert_rules (For Real-Time Alerting)
-- Define rules for suspicious patterns
-- ============================================================================

CREATE TABLE IF NOT EXISTS alert_rules (
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) CHECK (rule_type IN ('watchlist', 'frequency', 'pattern')),
    parameters JSONB,  -- Flexible rule parameters
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Example: Add a vehicle to watchlist
INSERT INTO alert_rules (rule_name, rule_type, parameters)
VALUES 
    ('Stolen Vehicle Watch', 'watchlist', '{"license_plates": ["MH12XX9999", "DL01YY8888"]}'),
    ('High Frequency Alert', 'frequency', '{"threshold": 10, "time_window_hours": 1}')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- TABLE: alerts (Generated Alerts)
-- ============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id BIGSERIAL PRIMARY KEY,
    rule_id INTEGER REFERENCES alert_rules(rule_id),
    sighting_id BIGINT REFERENCES vehicle_sightings(sighting_id),
    alert_timestamp TIMESTAMPTZ DEFAULT NOW(),
    alert_message TEXT,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(alert_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unacknowledged ON alerts(is_acknowledged) WHERE NOT is_acknowledged;

-- ============================================================================
-- TRIGGER: Auto-update timestamp on cameras table
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_cameras_updated_at 
BEFORE UPDATE ON cameras
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTION: Get vehicle history (common query)
-- ============================================================================

CREATE OR REPLACE FUNCTION get_vehicle_history(
    plate VARCHAR(15),
    days_back INTEGER DEFAULT 7
)
RETURNS TABLE (
    sighting_id BIGINT,
    camera_name VARCHAR(100),
    timestamp TIMESTAMPTZ,
    vehicle_color VARCHAR(20),
    direction VARCHAR(20),
    confidence REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        vs.sighting_id,
        c.camera_name,
        vs.timestamp,
        vs.vehicle_color,
        vs.direction,
        vs.confidence
    FROM vehicle_sightings vs
    JOIN cameras c ON vs.camera_id = c.camera_id
    WHERE vs.license_plate = plate
        AND vs.timestamp > NOW() - (days_back || ' days')::INTERVAL
    ORDER BY vs.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Example usage:
-- SELECT * FROM get_vehicle_history('MH12AB1234', 7);

-- ============================================================================
-- FUNCTION: Real-time traffic flow analysis
-- ============================================================================

CREATE OR REPLACE FUNCTION get_traffic_flow(
    cam_id INTEGER,
    time_window_minutes INTEGER DEFAULT 60
)
RETURNS TABLE (
    direction VARCHAR(20),
    vehicle_count BIGINT,
    avg_confidence REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        vs.direction,
        COUNT(*) AS vehicle_count,
        AVG(vs.confidence)::REAL AS avg_confidence
    FROM vehicle_sightings vs
    WHERE vs.camera_id = cam_id
        AND vs.timestamp > NOW() - (time_window_minutes || ' minutes')::INTERVAL
    GROUP BY vs.direction;
END;
$$ LANGUAGE plpgsql;

-- Example usage:
-- SELECT * FROM get_traffic_flow(1, 60);

-- ============================================================================
-- PARTITIONING: For high-volume deployments (optional)
-- Partition vehicle_sightings by month for better performance
-- ============================================================================

-- Uncomment below for production deployments with millions of records

/*
-- Drop existing table and recreate as partitioned
DROP TABLE IF EXISTS vehicle_sightings CASCADE;

CREATE TABLE vehicle_sightings (
    sighting_id BIGSERIAL,
    camera_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    license_plate VARCHAR(15) NOT NULL,
    vehicle_color VARCHAR(20),
    direction VARCHAR(20) CHECK (direction IN ('towards_camera', 'away_from_camera', 'unknown')),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    vehicle_type VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (sighting_id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for each month
CREATE TABLE vehicle_sightings_2025_01 PARTITION OF vehicle_sightings
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE vehicle_sightings_2025_02 PARTITION OF vehicle_sightings
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Add more partitions as needed...

-- Create indexes on each partition
CREATE INDEX idx_sightings_2025_01_timestamp ON vehicle_sightings_2025_01(timestamp DESC);
CREATE INDEX idx_sightings_2025_01_plate ON vehicle_sightings_2025_01(license_plate);
*/

-- ============================================================================
-- GRANTS: Security and access control
-- ============================================================================

-- Create roles for different access levels
CREATE ROLE analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_readonly;

CREATE ROLE analytics_readwrite;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO analytics_readwrite;

-- Grant sequence permissions for INSERT operations
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO analytics_readwrite;

-- Example: Create user and assign role
-- CREATE USER analytics_app WITH PASSWORD 'secure_password';
-- GRANT analytics_readwrite TO analytics_app;

-- ============================================================================
-- MAINTENANCE: Cleanup old data (optional)
-- ============================================================================

-- Delete sightings older than 90 days
-- Run this periodically via cron or pg_cron extension
CREATE OR REPLACE FUNCTION cleanup_old_sightings(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM vehicle_sightings
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Example usage:
-- SELECT cleanup_old_sightings(90);

-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

DO $$ 
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Database schema created successfully!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - cameras';
    RAISE NOTICE '  - vehicle_sightings';
    RAISE NOTICE '  - vehicle_journeys';
    RAISE NOTICE '  - alert_rules';
    RAISE NOTICE '  - alerts';
    RAISE NOTICE '';
    RAISE NOTICE 'Indexes created for optimal query performance';
    RAISE NOTICE 'Sample data inserted into cameras table';
    RAISE NOTICE '';
END $$;
