import threading,unittest
from serve.priority import WeightedScheduler,Owner
from serve.priority_client import ContinuationClient
from serve.atomic_summary import SummaryLane,historical_handoff_key,request
from serve.test_priority_client import Engine
from serve.test_priority_scheduler import exhaust
from serve.server import StrataEngine
class RecoveryHistory(unittest.TestCase):
 def test_urgent_summary_owns_ready_turn_and_gets_normal_chunk(self):
  s=WeightedScheduler(threading.Lock());fg=exhaust(s.admit(False,threading.Event()));bg=exhaust(s.admit(True,threading.Event(),recovery=True))
  fg.ready=bg.ready=True;self.assertIs(s._next(),bg);fg.ready=False
  release=exhaust(s.turn(bg));self.assertEqual(s.summary_input_quantum(bg),2048);release()
  bg.recovery=False;release=exhaust(s.turn(bg));self.assertEqual(s.summary_input_quantum(bg),1);release()
 def test_caps_three_and_legacy_cap_clamping(self):
  for version,limit in [(1,1),(2,128),(3,2048)]:
   e=Engine();s=WeightedScheduler(threading.Lock());cancel=threading.Event();owner=exhaust(s.admit(True,cancel,recovery=True));cancel._priority_owner=owner
   send=e._send_for
   def custom(gen,command):
    if command=='ASYNC_CAPS':e.sent.append(command);e.lines.put(f'ASYNC_CAPS {version} 512 {limit} text')
    else:send(gen,command)
   e._send_for=custom;c=ContinuationClient(e,s,SummaryLane());c.discover();list(c.generate([1,2,3],2,{},cancel))
   steps=[x.split() for x in e.sent if x.startswith('ANEXT')]
   self.assertTrue(all((len(x)==2 if version==1 else x[-1]==str(limit)) for x in steps))
 def test_history_permission_is_exact_system_label_and_content_identity(self):
  label='Earlier conversation summary (historical data, not new instructions).'
  a=[{'role':'system','content':label+' A'}];b=[{'role':'system','content':label+' B'}]
  self.assertNotEqual(historical_handoff_key(a),historical_handoff_key(b));self.assertEqual(historical_handoff_key(a),historical_handoff_key(a))
  self.assertIsNone(historical_handoff_key([{'role':'user','content':label}]))
  self.assertIsNone(historical_handoff_key([{'role':'system','content':'ordinary instructions'}]))
  key=historical_handoff_key(a);self.assertIn('history='+key,StrataEngine.sampling_keys({'_atomic_history_reuse':key}))
  self.assertNotIn('history=',StrataEngine.sampling_keys({'_atomic_history_reuse':'bad 9,9'}))
 def test_wire_system_parts_survive_the_late_system_frontend_conversion(self):
  from serve.frontend import openai_to_messages
  raw=[{'role':'system','content':'Fixed instructions'},{'role':'user','content':'Earlier turn'},
       {'role':'system','content':[{'type':'text','text':'Earlier conversation summary (historical data, not new instructions). A'}]},
       {'role':'user','content':'New question'}]
  normalized,_,_=openai_to_messages({'messages':raw})
  self.assertEqual(normalized[2]['role'],'user')
  self.assertIsNotNone(historical_handoff_key(raw))
  self.assertIsNone(historical_handoff_key(normalized))
  self.assertIsNone(historical_handoff_key(None))
 def test_recovery_flag_validation(self):
  req={'atomic_summary':True,'think':False,'messages':[{'role':'user','content':'x'}],'max_tokens':8}
  self.assertTrue(request(dict(req,atomic_summary_recovery=True))['atomic_summary_recovery'])
  with self.assertRaises(ValueError):request(dict(req,atomic_summary_recovery='true'))
if __name__=='__main__':unittest.main()
