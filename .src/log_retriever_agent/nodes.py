import paramiko
from datetime import datetime, timedelta, timezone
from state import LogAgentState 

def fetch_ssh_logs(state: LogAgentState):
    # 1. Generate a unique filename with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%d%m%Y_%H%M%S")
    state['log_file_path'] = f"/tmp/log_{state['host']}_{timestamp}.log"
    
    # 2. Initialize SSH Client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 3. Connect using credentials from the state
        client.connect(
            hostname=state['host'], 
            username=state['user'], 
            password=state['password'],
            timeout=10
        )

        # 4. Calculate the time window 
        minutes_to_fetch = 5
        start_time = (datetime.now(timezone.utc) - timedelta(minutes=minutes_to_fetch)).strftime('%Y-%m-%d %H:%M:%S')
        command = f"journalctl --since '{start_time}' --no-pager"

        # 5. Execute command and read output
        stdin, stdout, stderr = client.exec_command(command)
        logs = stdout.read().decode('utf-8')

   
        with open(state['log_file_path'], "w") as f:
            f.write(logs)

        print(f"--- Successfully saved logs to {state['log_file_path']} ---")
        state['status'] = "success"

    except Exception as e:
        print(f"--- [Log Agent] Error: {str(e)} ---")
        state['status'] = f"failed: {str(e)}"
    
    finally:
        client.close()
    
    return state
