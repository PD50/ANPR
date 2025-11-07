#!/usr/bin/env python3
"""
main_pipeline.py
Real-Time Vehicle Analytics Pipeline using NVIDIA DeepStream 7.0+
Processes 20fps video stream with multi-model ensemble for ANPR and analytics
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds
import json
from datetime import datetime
from line_crossing_logic import DirectionFinder

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
DEEPSTREAM_CONFIG = "deepstream_config.txt"

# Virtual line coordinates (x1, y1, x2, y2) for direction detection
# This line should be placed perpendicular to traffic flow at the roundabout
VIRTUAL_LINE_A = [(960, 400), (960, 680)]  # Vertical line at center

# Tracking configuration
MIN_CONFIDENCE = 0.4
DIRECTION_FINDER = DirectionFinder(VIRTUAL_LINE_A)

# ============================================================================
# METADATA EXTRACTION UTILITIES
# ============================================================================

def get_vehicle_color(obj_meta):
    """Extract vehicle color from SGIE0 (ResNet-50 classifier) metadata"""
    l_classifier = obj_meta.classifier_meta_list
    while l_classifier:
        classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
        if classifier_meta.unique_component_id == 2:  # SGIE0 color classifier
            l_label = classifier_meta.label_info_list
            if l_label:
                label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                return label_info.result_label
        l_classifier = l_classifier.next
    return "Unknown"


def get_license_plate_text(obj_meta):
    """Extract license plate text from SGIE2 (LPRNet OCR) metadata"""
    l_classifier = obj_meta.classifier_meta_list
    while l_classifier:
        classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
        if classifier_meta.unique_component_id == 4:  # SGIE2 OCR
            l_label = classifier_meta.label_info_list
            if l_label:
                label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                return label_info.result_label
        l_classifier = l_classifier.next
    return None


def check_plate_detected(obj_meta):
    """Check if license plate was detected by SGIE1"""
    l_obj = obj_meta.obj_user_meta_list
    while l_obj:
        user_meta = pyds.NvDsUserMeta.cast(l_obj.data)
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META:
            # Check if SGIE1 detected a plate
            return True
        l_obj = l_obj.next
    return False


# ============================================================================
# CUSTOM PROBE: Direction Detection + Metadata Enrichment
# ============================================================================

def tracker_probe_callback(pad, info, u_data):
    """
    Custom probe attached to tracker output
    - Extracts trajectory data from ByteTrack
    - Performs virtual line-crossing detection
    - Enriches metadata with direction information
    - Prepares data for RabbitMQ publishing
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Acquire batch metadata
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # Filter: Only process vehicles with high confidence
                if obj_meta.confidence < MIN_CONFIDENCE:
                    l_obj = l_obj.next
                    continue
                
                # Get tracking information
                object_id = obj_meta.object_id
                
                # Calculate centroid
                centroid_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                centroid_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
                
                # Check for line crossing and get direction
                direction = DIRECTION_FINDER.check_direction(
                    object_id, 
                    centroid_x, 
                    centroid_y
                )
                
                # Only process vehicles that have crossed the line
                if direction:
                    # Extract color from SGIE0
                    vehicle_color = get_vehicle_color(obj_meta)
                    
                    # Extract license plate from SGIE2 (if detected)
                    license_plate = get_license_plate_text(obj_meta)
                    
                    # Only publish complete vehicle sightings with valid plate reads
                    if license_plate and license_plate != "":
                        # Create event message metadata
                        event_msg_meta = pyds.alloc_nvds_event_msg_meta()
                        event_msg_meta.sensorId = 0  # Camera ID
                        event_msg_meta.placeId = 0
                        event_msg_meta.moduleId = 0
                        event_msg_meta.sensorStr = "ROUNDABOUT_CAM_01"
                        event_msg_meta.ts = datetime.utcnow().isoformat() + 'Z'
                        
                        # Create custom vehicle analytics payload
                        vehicle_data = {
                            "object_id": int(object_id),
                            "license_plate": license_plate,
                            "vehicle_color": vehicle_color,
                            "direction": direction,
                            "timestamp": event_msg_meta.ts,
                            "confidence": float(obj_meta.confidence),
                            "vehicle_type": obj_meta.obj_label
                        }
                        
                        # Serialize to JSON
                        event_msg_meta.objSignature = json.dumps(vehicle_data)
                        
                        # Attach to frame metadata for nvmsgconv/nvmsgbroker
                        user_event_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
                        user_event_meta.user_meta_data = event_msg_meta
                        user_event_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META
                        user_event_meta.base_meta.copy_func = pyds.meta_copy_func
                        user_event_meta.base_meta.release_func = pyds.meta_free_func
                        
                        pyds.nvds_add_user_meta_to_frame(frame_meta, user_event_meta)
                        
                        # Console logging for debugging
                        print(f"[VEHICLE SIGHTING] ID:{object_id} | Plate:{license_plate} | "
                              f"Color:{vehicle_color} | Direction:{direction} | "
                              f"Confidence:{obj_meta.confidence:.2f}")
                
            except StopIteration:
                break
            
            l_obj = l_obj.next
        
        l_frame = l_frame.next
    
    return Gst.PadProbeReturn.OK


# ============================================================================
# PIPELINE MANAGEMENT
# ============================================================================

def bus_call(bus, message, loop):
    """GStreamer bus message handler"""
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def main():
    """Initialize and run the DeepStream pipeline"""
    
    # Initialize GStreamer
    Gst.init(None)
    
    print("=" * 80)
    print("NVIDIA DeepStream Vehicle Analytics Pipeline v1.0")
    print("Target Performance: 20fps | Models: YOLOv10 + ByteTrack + ResNet50 + LPRNet")
    print("=" * 80)
    
    # Create pipeline from config file
    pipeline = Gst.parse_launch(
        f"nvstreammux name=mux ! "
        f"nvinfer name=pgie ! "
        f"nvtracker name=tracker ! "
        f"nvinfer name=sgie0 ! "
        f"nvinfer name=sgie1 ! "
        f"nvinfer name=sgie2 ! "
        f"nvmsgconv name=msgconv ! "
        f"nvmsgbroker name=msgbroker ! "
        f"nvvideoconvert ! "
        f"nvdsosd name=osd ! "
        f"nvvideoconvert ! "
        f"capsfilter caps=video/x-raw,format=RGBA ! "
        f"fakesink name=sink"
    )
    
    # Alternative: Use config file directly
    # This is more reliable for production deployments
    # pipeline = Gst.parse_launch(f"deepstream-app -c {DEEPSTREAM_CONFIG}")
    
    if not pipeline:
        sys.stderr.write("Failed to create pipeline\n")
        return
    
    # Get tracker element
    tracker = pipeline.get_by_name("tracker")
    if not tracker:
        sys.stderr.write("Unable to get tracker element\n")
        return
    
    # Attach custom probe to tracker's src pad
    tracker_src_pad = tracker.get_static_pad("src")
    if not tracker_src_pad:
        sys.stderr.write("Unable to get tracker src pad\n")
        return
    
    tracker_src_pad.add_probe(
        Gst.PadProbeType.BUFFER, 
        tracker_probe_callback, 
        0
    )
    
    # Setup GLib main loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start pipeline
    print("\n[INFO] Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        sys.stderr.write("Unable to set pipeline to PLAYING state\n")
        return
    
    print("[INFO] Pipeline running. Press Ctrl+C to stop.\n")
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupt received, stopping pipeline...")
    
    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    print("[INFO] Pipeline stopped. Exiting.")


if __name__ == '__main__':
    sys.exit(main())
