"""Measure quantization with synthetic LiDAR; never exposes walls to a policy."""
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from util import OnlineOccupancyMap as Map, FrontierTeacher
from sim2d_env import make_layout, raycast


def connected(mapping,start,end):
    free=FrontierTeacher.planning_free(mapping.snapshot())
    distance,_=FrontierTeacher.search(free,tuple(Map.cells(start)))
    return bool(np.isfinite(distance[tuple(Map.cells(end))]))


def measure():
    results=[]
    angles=np.arange(360)*2*math.pi/360
    directions=np.stack((np.cos(angles),np.sin(angles)),-1)
    # A straight corridor and a transverse door; dimensions/alignments are independent of production scenes.
    for kind in ('corridor','door'):
        for width in (.3,.4,.6,.8,1.2):
            for offset in (0.,.1):
                walls=[[-6.1,-6.1,-5.9,6.1],[5.9,-6.1,6.1,6.1],[-5.9,-6.1,5.9,-5.9],[-5.9,5.9,5.9,6.1]]
                if kind=='corridor':
                    walls += [[-5.9,-5.9,5.9,offset-width/2],[-5.9,offset+width/2,5.9,5.9]]
                else:
                    walls += [[-.1,-5.9,.1,offset-width/2],[-.1,offset+width/2,.1,5.9]]
                walls=np.array(walls)
                mapping=Map(0.,2*math.pi/360)
                # Scan from both sides, then inspect the shared observed map (not geometry).
                for x in np.arange(-2.,2.01,.2):
                    position=np.array([x,offset])
                    ranges=raycast(position-[.032,0.],directions,walls)
                    mapping.observe(ranges,position,0.)
                results.append(dict(kind=kind,width=width,offset=offset,
                                    sensor_map_connected=connected(mapping,[-2.,offset],[2.,offset])))
    # The first four fixed validation narrow tasks: inspect observations along the actual corridor skeleton.
    for seed in (5000004,5000009,5000014,5000019):
        walls,start,goal,yaw=make_layout('narrow',seed)
        mapping=Map(0.,2*math.pi/360)
        rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        points=np.concatenate((np.column_stack((np.arange(-4.6,1.01,.2),np.full(29,-2.))),
                               np.column_stack((np.ones(21),np.linspace(-2.,2.,21))),
                               np.column_stack((np.linspace(1.,4.6,19),np.full(19,2.))))) @ rotation.T
        for position in points:
            ranges=raycast(position-[.032,0.],directions,walls)
            mapping.observe(ranges,position,0.)
        results.append(dict(kind='narrow_validation',seed=seed,sensor_map_connected=connected(mapping,start,goal)))
    return results


if __name__=='__main__':
    print(json.dumps(measure(),indent=2))
