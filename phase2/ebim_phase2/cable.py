"""Phase II route accounting from explicit fresh perception, without old fixture coordinates."""
from __future__ import annotations
from dataclasses import dataclass
import math
from numbers import Integral
import time


@dataclass(frozen=True)
class Route:
    route_id: str
    fixtures: tuple[str, ...]
    directions: tuple[str, ...]

    def __post_init__(self):
        if not self.route_id or not self.fixtures or len(self.fixtures)!=len(self.directions):
            raise ValueError("A route requires an ID and ordered fixture/direction pairs")
        if len(set(self.fixtures))!=len(self.fixtures):
            raise ValueError("Fixture identifiers must be unique")
        if any(not isinstance(x,str) or not x.strip() for x in (*self.fixtures,*self.directions)):
            raise ValueError('Fixture identifiers and assigned directions must be nonempty strings')


def score_prefix(route: Route, correct: dict[str, bool], validated_through: int) -> dict:
    """Checkpoint authority comes from the current problem's verifier, never a guessed old layout.

    validated_through is a fixture count, supplied by the checkpoint verifier; it does not
    repair an incorrect prefix. Set it to the total fixture count only when the applicable
    current rules/verifier validate the whole route.
    """
    if isinstance(validated_through,bool) or not isinstance(validated_through,Integral) or not 0<=validated_through<=len(route.fixtures):
        raise ValueError("Invalid checkpoint boundary")
    if set(correct)-set(route.fixtures) or any(type(value) is not bool for value in correct.values()):
        raise ValueError('Correctness observations must map assigned fixture IDs to booleans')
    prefix=0
    for fixture in route.fixtures:
        if correct.get(fixture) is not True:break
        prefix+=1
    credited=min(prefix,validated_through)
    return {"correct_prefix":prefix,"validated_prefix":credited,"total_fixtures":len(route.fixtures),
            "completion_percent":100*credited/len(route.fixtures)}


class RouteMonitor:
    def __init__(self,route:Route,max_age_s=.2,stall_timeout_s=30.):
        if not all(math.isfinite(x) and x>0 for x in (max_age_s,stall_timeout_s)):
            raise ValueError('Progress age and stall limits must be finite and positive')
        self.route=route;self.max_age=max_age_s;self.stall_timeout=stall_timeout_s
        self.started=None;self.last_progress=None;self.best_prefix=0;self.best_validated=0
        self.last_timestamp=None;self.last_clock=None
        self.last_observation=None

    def update(self,*,route_id,timestamp_s,now_s,correct,validated_through):
        if route_id!=self.route.route_id:
            raise ValueError("Route changed; reset policy and monitor with the new assignment")
        if not all(math.isfinite(x) for x in (timestamp_s,now_s)) or not 0<=now_s-timestamp_s<=self.max_age:
            return {"status":"stop","reason":"stale_route_perception"}
        if (self.last_timestamp is not None and timestamp_s<self.last_timestamp) or (self.last_clock is not None and now_s<=self.last_clock):
            return {'status':'stop','reason':'nonmonotonic_route_perception'}
        observed=(dict(correct),validated_through)
        if timestamp_s==self.last_timestamp and observed!=self.last_observation:
            return {'status':'stop','reason':'changed_route_evidence_at_same_timestamp'}
        if self.started is None:self.started=self.last_progress=now_s
        if now_s-self.started>=1800:return {"status":"stop","reason":"task1_time_limit"}
        score=score_prefix(self.route,correct,validated_through)
        self.last_timestamp,self.last_clock=timestamp_s,now_s
        self.last_observation=observed
        prefix=score['correct_prefix']
        if prefix>self.best_prefix:self.best_prefix=prefix;self.last_progress=now_s
        if score['validated_prefix']>self.best_validated:
            self.best_validated=score['validated_prefix'];self.last_progress=now_s
        status='complete' if score['validated_prefix']==len(self.route.fixtures) else 'routing'
        if prefix<self.best_prefix or score['validated_prefix']<self.best_validated:status='revalidate_prefix'
        elif status!='complete' and now_s-self.last_progress>=self.stall_timeout:status='recovery_required'
        elif prefix==len(self.route.fixtures) and status!='complete':status='awaiting_checkpoint'
        return {**score,'status':status,'next_fixture':self.route.fixtures[prefix] if prefix<len(self.route.fixtures) else None,
                'next_direction':self.route.directions[prefix] if prefix<len(self.route.fixtures) else None}


class SupervisedRoutingPolicy:
    """Gate a learned Task1 proposal with the current route's measured progress.

    This supervisor does not turn a route-agnostic ACT checkpoint into a
    route-conditioned model. It exposes the next assignment, suppresses actions
    after regression/stall/completion, and requires a fresh independent verifier.
    A correctness flag means both fixture engagement AND assigned direction.
    """
    def __init__(self,policy,route:Route):
        if policy.task!=1:raise ValueError('Route supervision requires a Task1 checkpoint')
        self.policy=policy;self.route=route
        self.task=policy.task;self.limiter=policy.limiter;self.ensemble=policy.ensemble
        self.episode_id=None;self.started=None;self.monitor=RouteMonitor(route)

    def infer(self,record,*,now=None):
        if not isinstance(record,dict):
            raise ValueError('Observation must be a JSON object')
        clock=time.time() if now is None else float(now)
        if not math.isfinite(clock):raise ValueError('Invalid receiver clock')
        episode=record.get('episode_id')
        if episode is None:raise ValueError('episode_id is required')
        if episode!=self.episode_id:
            self.episode_id=episode;self.started=clock;self.monitor=RouteMonitor(self.route)
            self.policy.reset(episode)
        if clock-self.started>=1800:
            progress={'status':'stop','reason':'task1_time_limit'}
        else:
            observed=record.get('route_progress')
            if not isinstance(observed,dict):
                progress={'status':'stop','reason':'missing_route_perception'}
            elif observed.get('source') not in ('vision','rgbd','tactile') or not .8<=float(observed.get('confidence',0))<=1:
                progress={'status':'stop','reason':'uncertain_route_perception'}
            else:
                progress=self.monitor.update(route_id=observed['route_id'],timestamp_s=float(observed['timestamp_s']),
                    now_s=clock,correct=observed['correct'],validated_through=observed['validated_through'])
        if progress['status']=='routing':
            result=self.policy.infer(record,now=now)
        else:
            self.ensemble.reset()
            result={'team':'AirSign','task':1,'episode_id':episode,'observation_timestamp_s':record.get('timestamp_s'),
                    'status':'no_command','reason':progress.get('reason',progress['status']),'action':None,'commands':None}
        result['route']={'route_id':self.route.route_id,**progress}
        return result
