# We used North Central US as the region for this
import neuropythy as ny
_BATCH_ACCOUNT_NAME = 'XXX'  # Your batch account name
_BATCH_ACCOUNT_KEY = 'XXX'  # Your batch account key
# Your batch account URL
_BATCH_ACCOUNT_URL = 'XXX'
_STORAGE_ACCOUNT_NAME = 'XXX'  # Your storage account name
# Your storage account key
_STORAGE_ACCOUNT_KEY = 'XXX'

_POOL_NODE_COUNT = 192  # Pool node count
_POOL_VM_SIZE = 'STANDARD_E8_v3'  # VM Type/Size
_POOL_ID = 'XXX'  # Your Pool ID
_JOB_ID = 'XXX'  # Job ID
_STANDARD_OUT_FILE_NAME = 'stdout.txt'  # Standard Output file
_STANDARD_ERR_FILE_NAME = 'stderr.txt'  # Standard Error file

_TASK_FILE = "task_dki_hcp_variability.py"
_OUTBUCKET = "hcp.dki"

_TIMEOUT = 1440 * 2

SUBJECTS = ny.hcp.subject_ids
