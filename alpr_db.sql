CREATE DATABASE IF NOT EXISTS alpr_db;
USE alpr_db;

-- Vehicles table (master list of unique plates)
CREATE TABLE IF NOT EXISTS Vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_text VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Main Events table (one row per detection)
CREATE TABLE IF NOT EXISTS Events (
    event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    plate_text VARCHAR(50) NOT NULL,                          -- Added missing column
    event_type ENUM('Entry', 'Exit') NOT NULL DEFAULT 'Entry', -- Default so Python insert works even if not passed yet
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    picture_path VARCHAR(255) NOT NULL,
    confidence float DEFAULT NULL,
    gate_number SMALLINT NOT NULL CHECK (gate_number BETWEEN 1 AND 6),
    
    FOREIGN KEY (plate_text) REFERENCES Vehicles(plate_text) ON DELETE CASCADE,
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_plate_time (plate_text, timestamp),
    INDEX idx_gate (gate_number)
);

-- Archive table (same structure as Events for easy copying)
CREATE TABLE IF NOT EXISTS Events_Archive (
    event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    plate_text VARCHAR(50) NOT NULL,
    event_type ENUM('Entry', 'Exit') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    picture_path VARCHAR(255) NOT NULL,
    confidence float DEFAULT NULL,
    gate_number SMALLINT NOT NULL CHECK (gate_number BETWEEN 1 AND 6),
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_plate_time (plate_text, timestamp),  -- Fixed typo (dettection_time → detection_time)
    INDEX idx_gate (gate_number)
);

-- Enable event scheduler
SET GLOBAL event_scheduler = ON;

-- Daily archiving event (safer INSERT without using SELECT *)
DELIMITER $$

CREATE EVENT archive_old_events
    ON SCHEDULE EVERY 30 day
    STARTS CURRENT_TIMESTAMP + INTERVAL 2 HOUR  -- Next ~uvan 2 AM
    ON COMPLETION PRESERVE
    ENABLE
    COMMENT 'Archives and removes Events older than 30 days'
DO
BEGIN
    -- Archive old events (explicit columns = safer, new auto-increment ID)
    INSERT INTO Events_Archive 
        (plate_text, event_type, timestamp, picture_path, confidence, gate_number)
    SELECT 
        plate_text, event_type, timestamp, picture_path, confidence, gate_number
    FROM Events
    WHERE timestamp < NOW() - INTERVAL 30 DAY;

    -- Delete archived events from main table
    DELETE FROM Events
    WHERE timestamp < NOW() - INTERVAL 30 DAY;
END$$

DELIMITER ;

