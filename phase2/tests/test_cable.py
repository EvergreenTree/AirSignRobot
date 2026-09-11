from ebim_phase2.cable import Route,RouteMonitor,score_prefix
from ebim_phase2.cable import SupervisedRoutingPolicy
import pytest


def test_incorrect_prefix_cannot_be_bypassed_by_later_checkpoint():
    route=Route('phase2-board',('a','b','c'),('cw','through','ccw'))
    assert score_prefix(route,{'a':True,'b':False,'c':True},3)['completion_percent']==100/3
    assert score_prefix(route,{'a':True,'b':True,'c':True},1)['validated_prefix']==1


def test_monitor_requires_fresh_observed_progress():
    route=Route('new',('a','b'),('cw','ccw'));m=RouteMonitor(route,stall_timeout_s=1)
    assert m.update(route_id='new',timestamp_s=0,now_s=1,correct={},validated_through=0)['status']=='stop'
    m.update(route_id='new',timestamp_s=1,now_s=1,correct={},validated_through=0)
    assert m.update(route_id='new',timestamp_s=2.1,now_s=2.1,correct={},validated_through=0)['status']=='recovery_required'
    assert m.update(route_id='new',timestamp_s=3,now_s=3,correct={'a':True,'b':True},validated_through=2)['status']=='complete'


def test_supervisor_stops_on_regression_and_requires_checkpoint():
    class Stub:
        task=1;limiter=None
        class Ensemble:
            def reset(self):pass
        ensemble=Ensemble()
        def reset(self,episode):pass
        def infer(self,record,now=None):return {'status':'shadow_native_proposal','action':None}
    policy=SupervisedRoutingPolicy(Stub(),Route('current',('a','b'),('cw','through')))
    def observe(t,correct,boundary):
        return policy.infer({'episode_id':'trial','timestamp_s':t,'route_progress':{
            'route_id':'current','timestamp_s':t,'source':'vision','confidence':.95,
            'correct':correct,'validated_through':boundary}},now=t)
    assert policy.infer({'episode_id':'trial'},now=0)['reason']=='missing_route_perception'
    first=observe(1,{'a':True},1)
    assert first['route']['next_fixture']=='b' and first['route']['next_direction']=='through'
    assert observe(2,{},0)['reason']=='revalidate_prefix'
    assert observe(3,{'a':True,'b':True},1)['reason']=='awaiting_checkpoint'
    assert observe(4,{'a':True,'b':True},2)['reason']=='complete'
    assert observe(4,{'a':True,'b':True},2)['reason']=='nonmonotonic_route_perception'
    assert observe(1800,{},0)['reason']=='task1_time_limit'


def test_checkpoint_boundaries_and_flags_are_explicit():
    route=Route('current',('a',),('cw',))
    with pytest.raises(ValueError):score_prefix(route,{'a':True},.5)
    with pytest.raises(ValueError):score_prefix(route,{'a':.9},1)


def test_fresh_route_evidence_can_be_reused_between_slower_vision_updates():
    monitor=RouteMonitor(Route('board',('a','b'),('cw','through')))
    evidence={'route_id':'board','timestamp_s':1.,'correct':{'a':True},'validated_through':1}
    assert monitor.update(**evidence,now_s=1.)['status']=='routing'
    assert monitor.update(**evidence,now_s=1.05)['status']=='routing'
    changed=monitor.update(**{**evidence,'correct':{'a':True,'b':True}},now_s=1.10)
    assert changed['reason']=='changed_route_evidence_at_same_timestamp'
    assert monitor.update(**evidence,now_s=1.3)['reason']=='stale_route_perception'
