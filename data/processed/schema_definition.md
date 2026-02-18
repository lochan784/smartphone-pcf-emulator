PRIMARY FEATURES (Structured)

brand                 (string)
model                 (string)
year                  (int)

battery_mah           (float)
screen_size_in        (float)
display_area_cm2      (float, engineered)

ram_gb                (float)
storage_gb            (float)

performance_index     (float, engineered)
rear_camera_count     (int)
total_camera_mp       (float)
refresh_rate_hz       (float)

estimated_mass_g      (float, engineered)
mass_estimated        (bool)

SEMANTIC FEATURE

spec_text             (string)

LABELS

synthetic_label       (bool)
pcf_kg_co2e           (float, nullable)

AUDIT FIELDS

source_url            (string)
date_collected        (date)
license               (string)
