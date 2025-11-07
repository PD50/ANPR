#!/usr/bin/env python3
"""
db_consumer.py
Asynchronous database consumer for vehicle analytics pipeline
Subscribes to RabbitMQ and persists vehicle sightings to PostgreSQL
"""

import json
import sys
import signal
import time
from datetime import datetime
from typing import Dict, Optional

import pika
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import pool


# ============================================================================
# CONFIGURATION
# ============================================================================

# RabbitMQ Configuration
RABBITMQ_CONFIG = {
    'host': 'localhost',
    'port': 5672,
    'username': 'guest',
    'password': 'guest',
    'queue': 'vehicle_sightings',
    'exchange': '',
    'routing_key': 'vehicle_sightings',
    'prefetch_count': 10  # Process N messages at a time
}

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vehicle_analytics',
    'user': 'postgres',
    'password': 'your_password',
    'minconn': 2,
    'maxconn': 10
}

# Camera mapping (for multi-camera deployments)
CAMERA_REGISTRY = {
    'ROUNDABOUT_CAM_01': 1,  # Maps sensor string to camera_id
}

# Operational parameters
BATCH_INSERT_SIZE = 50  # Batch inserts for performance
BATCH_TIMEOUT_SECONDS = 5.0  # Max wait before flushing batch


# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================

class DatabaseManager:
    """Manages PostgreSQL connection pool and batch operations"""
    
    def __init__(self, config: Dict):
        """Initialize connection pool"""
        self.config = config
        self.pool = None
        self.pending_batch = []
        self.last_flush_time = time.time()
        
        self._initialize_pool()
    
    
    def _initialize_pool(self):
        """Create connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config['minconn'],
                maxconn=self.config['maxconn'],
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            print(f"[DB] Connection pool created: {self.config['minconn']}-{self.config['maxconn']} connections")
        except Exception as e:
            print(f"[ERROR] Failed to create connection pool: {e}")
            sys.exit(1)
    
    
    def get_connection(self):
        """Acquire connection from pool"""
        return self.pool.getconn()
    
    
    def return_connection(self, conn):
        """Return connection to pool"""
        self.pool.putconn(conn)
    
    
    def add_to_batch(self, record: tuple):
        """Add record to pending batch"""
        self.pending_batch.append(record)
    
    
    def should_flush_batch(self) -> bool:
        """Check if batch should be flushed"""
        if len(self.pending_batch) >= BATCH_INSERT_SIZE:
            return True
        if time.time() - self.last_flush_time > BATCH_TIMEOUT_SECONDS:
            return True
        return False
    
    
    def flush_batch(self) -> int:
        """Flush pending batch to database"""
        if not self.pending_batch:
            return 0
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Use execute_values for fast batch insert
            insert_query = """
                INSERT INTO vehicle_sightings 
                (camera_id, timestamp, license_plate, vehicle_color, direction, confidence, vehicle_type)
                VALUES %s
                ON CONFLICT DO NOTHING
            """
            
            execute_values(cursor, insert_query, self.pending_batch)
            conn.commit()
            
            count = len(self.pending_batch)
            print(f"[DB] ✓ Flushed {count} records to database")
            
            # Clear batch
            self.pending_batch.clear()
            self.last_flush_time = time.time()
            
            cursor.close()
            return count
            
        except Exception as e:
            print(f"[ERROR] Batch insert failed: {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if conn:
                self.return_connection(conn)
    
    
    def close(self):
        """Flush remaining records and close pool"""
        self.flush_batch()
        if self.pool:
            self.pool.closeall()
            print("[DB] Connection pool closed")


# ============================================================================
# MESSAGE PROCESSING
# ============================================================================

def parse_vehicle_message(message_body: bytes) -> Optional[Dict]:
    """
    Parse JSON message from RabbitMQ
    
    Expected format:
    {
        "object_id": 123,
        "license_plate": "MH12AB1234",
        "vehicle_color": "Red",
        "direction": "towards_camera",
        "timestamp": "2025-01-15T10:30:45.123Z",
        "confidence": 0.92,
        "vehicle_type": "car"
    }
    """
    try:
        # Decode and parse JSON
        data = json.loads(message_body.decode('utf-8'))
        
        # Validate required fields
        required_fields = ['license_plate', 'vehicle_color', 'direction', 'timestamp']
        if not all(field in data for field in required_fields):
            print(f"[WARN] Message missing required fields: {data}")
            return None
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Message parsing failed: {e}")
        return None


def process_vehicle_sighting(data: Dict, db_manager: DatabaseManager):
    """
    Process vehicle sighting and add to database batch
    
    Args:
        data: Parsed vehicle sighting data
        db_manager: Database manager instance
    """
    try:
        # Extract fields
        camera_id = CAMERA_REGISTRY.get('ROUNDABOUT_CAM_01', 1)  # Default to camera 1
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        license_plate = data['license_plate'][:15]  # Enforce VARCHAR(15) limit
        vehicle_color = data['vehicle_color'][:20]
        direction = data['direction'][:20]
        confidence = data.get('confidence', 0.0)
        vehicle_type = data.get('vehicle_type', 'unknown')[:20]
        
        # Create record tuple
        record = (
            camera_id,
            timestamp,
            license_plate,
            vehicle_color,
            direction,
            confidence,
            vehicle_type
        )
        
        # Add to batch
        db_manager.add_to_batch(record)
        
        # Check if batch should be flushed
        if db_manager.should_flush_batch():
            db_manager.flush_batch()
        
        # Console logging
        print(f"[MSG] Processed: {license_plate} | {vehicle_color} | {direction}")
        
    except Exception as e:
        print(f"[ERROR] Failed to process sighting: {e}")


# ============================================================================
# RABBITMQ CONSUMER
# ============================================================================

class VehicleAnalyticsConsumer:
    """RabbitMQ consumer for vehicle analytics messages"""
    
    def __init__(self, rabbitmq_config: Dict, db_manager: DatabaseManager):
        """Initialize consumer"""
        self.config = rabbitmq_config
        self.db_manager = db_manager
        self.connection = None
        self.channel = None
        self.should_stop = False
    
    
    def connect(self):
        """Establish RabbitMQ connection"""
        try:
            credentials = pika.PlainCredentials(
                self.config['username'],
                self.config['password']
            )
            
            parameters = pika.ConnectionParameters(
                host=self.config['host'],
                port=self.config['port'],
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare queue (idempotent)
            self.channel.queue_declare(
                queue=self.config['queue'],
                durable=True  # Survive broker restart
            )
            
            # Set QoS - prefetch count
            self.channel.basic_qos(prefetch_count=self.config['prefetch_count'])
            
            print(f"[MQ] Connected to RabbitMQ: {self.config['host']}:{self.config['port']}")
            print(f"[MQ] Listening on queue: {self.config['queue']}")
            
        except Exception as e:
            print(f"[ERROR] RabbitMQ connection failed: {e}")
            sys.exit(1)
    
    
    def on_message(self, channel, method, properties, body):
        """Callback for incoming messages"""
        try:
            # Parse message
            data = parse_vehicle_message(body)
            
            if data:
                # Process and add to database batch
                process_vehicle_sighting(data, self.db_manager)
                
                # Acknowledge message
                channel.basic_ack(delivery_tag=method.delivery_tag)
            else:
                # Reject invalid messages
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                
        except Exception as e:
            print(f"[ERROR] Message processing error: {e}")
            # Negative acknowledgment - message goes back to queue
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    
    def start_consuming(self):
        """Start consuming messages"""
        try:
            self.channel.basic_consume(
                queue=self.config['queue'],
                on_message_callback=self.on_message,
                auto_ack=False  # Manual acknowledgment
            )
            
            print("[MQ] Starting message consumption...")
            print("[MQ] Press Ctrl+C to stop\n")
            
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            print("\n[INFO] Interrupt received, stopping consumer...")
            self.stop()
    
    
    def stop(self):
        """Stop consuming and close connection"""
        self.should_stop = True
        
        if self.channel:
            self.channel.stop_consuming()
        
        # Flush any pending database writes
        if self.db_manager:
            self.db_manager.flush_batch()
        
        if self.connection:
            self.connection.close()
            print("[MQ] Connection closed")


# ============================================================================
# MAIN
# ============================================================================

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n[INFO] Shutdown signal received...")
    sys.exit(0)


def main():
    """Main entry point"""
    print("=" * 80)
    print("Vehicle Analytics Database Consumer v1.0")
    print("RabbitMQ → PostgreSQL Pipeline")
    print("=" * 80)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize database manager
    print("\n[1/2] Initializing database connection pool...")
    db_manager = DatabaseManager(POSTGRES_CONFIG)
    
    # Initialize RabbitMQ consumer
    print("[2/2] Connecting to RabbitMQ...")
    consumer = VehicleAnalyticsConsumer(RABBITMQ_CONFIG, db_manager)
    consumer.connect()
    
    # Start consuming
    try:
        consumer.start_consuming()
    except Exception as e:
        print(f"[ERROR] Consumer crashed: {e}")
    finally:
        # Cleanup
        consumer.stop()
        db_manager.close()
        print("\n[INFO] Consumer stopped cleanly")


if __name__ == '__main__':
    main()
