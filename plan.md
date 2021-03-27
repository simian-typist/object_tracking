# Object Tracking

Goals:
    1. Track multiple people

## Variation

Binary Categories:

    * Online vs Batch
    * Multi camera vs Single Camera
    * Multi object vs Single Object

## Components

    * Detector - Detect object(s)
    * Predictor - Where's our next track likely to be. Kalman filter etc
    * Data assocation algorithm - Which objects match which tracks
    * Similarity Metrics - How we objects with old tracks
        * Distance - How far away is our detection from our prediction
        * Appearance similarity (histograms or person reid etc)
        * Overlap (Intersection over Union)
    * Storage
    * Clean up.

# Next steps

    * Use smaller yolo
    * Try different detectors
    * Limit detections to just people
    * Get better test material
        * Look at actual MOT stuff
        * Just grab videos off youtube
