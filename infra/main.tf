provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

resource "oci_database_autonomous_database" "fitfind" {
  compartment_id             = var.compartment_id
  db_name                    = "FITNESSDB"
  display_name               = "fitfind-db"
  admin_password             = var.admin_password
  cpu_core_count             = 1
  data_storage_size_in_tbs   = 1
  db_workload                = "OLTP"
  is_free_tier               = true
  is_auto_scaling_enabled    = false
}

output "oracle_db_connection_info" {
  value = {
    high         = oci_database_autonomous_database.fitfind.connection_strings[0].high
    medium       = oci_database_autonomous_database.fitfind.connection_strings[0].medium
    low          = oci_database_autonomous_database.fitfind.connection_strings[0].low
  }
}