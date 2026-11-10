import io, contextlib, sys, time
from hybrid_spatial_index import HybridTokyoGraph
from swarm_simulation_dynamic import DynamicSwarmSimulation
import agent_snn_v3 as A

d=sys.argv[1]; seed=int(sys.argv[2])
st={"moves":0,"disabled_edge":0,"node_gt06":0}
orig=A.SNNAgent._execute_move
def patched(self,nxt,cp,gp):
    g=self.graph; st["moves"]+=1
    if not g.is_edge_passable(self.current_node,nxt): st["disabled_edge"]+=1
    if g.get_node_damage(nxt)>0.6: st["node_gt06"]+=1
    return orig(self,nxt,cp,gp)
A.SNNAgent._execute_move=patched
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    g=HybridTokyoGraph("tokyo_graph.json"); g.build_index()
    sim=DynamicSwarmSimulation(g,num_agents=10,disaster_type=d,seed=seed)
    h0=sum(1 for v in g._node_damage.values() if v>0.6); e0=len(g._disabled_edges)
    t0=time.time(); sim.run(steps=200); dt=time.time()-t0
m=sim.adaptation_metrics()
hard={n for n,v in g._node_damage.items() if v>0.6}
sm=sim.shared_memory["blocked_nodes"]
unobserved=[n for n in sm if n not in sim.detection_log]
reasons={}
for a in sim.agents:
    if not a.alive and not a.reached_goal: reasons[a.death_reason]=reasons.get(a.death_reason,0)+1
print(f"{d} seed={seed} success={m['success_rate']} dead={reasons} reroutes={m['total_reroute_events']} lag={m['mean_adaptation_lag']} sim_time={dt:.1f}s")
print(f"   damage t=0: hard={h0} disabled={e0} | end: hard={len(hard)} damaged={len(g._node_damage)}/{len(g._nodes)} disabled={len(g._disabled_edges)}")
print(f"   moves={st['moves']} onto_disabled_edge={st['disabled_edge']} onto_node>0.6={st['node_gt06']}")
print(f"   shared blocked={len(sm)} (never observed by an agent: {len(unobserved)}) | truth hard nodes known to swarm: {len(sm&hard)}/{len(hard)}")
