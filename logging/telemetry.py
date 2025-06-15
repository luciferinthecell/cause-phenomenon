
import os, json, threading, time
from datetime import datetime

class TelemetryCollector:
    def __init__(self, dir='telemetry'):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)
        self._buf=[]
        self._lock=threading.Lock()
    def record_event(self,event,data=None):
        ev={'ts':datetime.utcnow().isoformat(),'event':event,'data':data or {}}
        with self._lock:
            self._buf.append(ev)
    def flush(self):
        with self._lock:
            if not self._buf: return None
            fname='tele_'+datetime.utcnow().strftime('%Y%m%dT%H%M%S')+'.json'
            path=os.path.join(self.dir,fname)
            with open(path,'w',encoding='utf-8') as f:
                json.dump(self._buf,f,ensure_ascii=False,indent=2)
            self._buf=[]
            return path
