import json
import socket
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "data" / "workers.json", encoding="utf-8-sig") as f:
    workers = json.load(f)["workers"]

HOST, PORT = "0.0.0.0", 8765
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(5)

print("SAFE-X WORKER IDENTIFICATION")
print(f"Loaded {len(workers)} worker profiles")
print(f"Listening on port {PORT}")

while True:
    conn, addr = server.accept()
    print(f"NodeMCU connected: {addr}")
    with conn:
        buffer = ""
        while True:
            data = conn.recv(1024)
            print("RAW RECEIVED:", repr(data), flush=True)
            if not data:
                break
            buffer += data.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                uid = line.strip().upper()
                if not uid:
                    continue
                print(f"\n[{datetime.now():%H:%M:%S}] Card: {uid}")
                worker = workers.get(uid)
                if worker:
                    print("IDENTITY: REGISTERED")
                    print(f"Employee ID: {worker['employee_id']}")
                    print(f"Name: {worker['name']}")
                    print(f"Designation: {worker['designation']}")
                    print("Required PPE: " + ", ".join(worker["required_ppe"]))
                    print("PPE DECISION: AWAITING CAMERA INSPECTION")
                else:
                    print("IDENTITY: UNKNOWN CARD")
                    print("ACCESS: DENY / MANUAL REVIEW")
                print("-" * 40)
    print("NodeMCU disconnected")
