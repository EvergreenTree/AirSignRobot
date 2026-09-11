"""Robot-component math on a synthetic chain; not a competition environment."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from ebim_phase2.mujoco_backend import RobotKinematics, ActuatorDiagnostic
from ebim_phase2.tasks.geometry import Pose, pose_error


def fixture(tmp_path):
    bodies, actuators, mapping = [], [], {'arm_actuator_mode': 'velocity', 'extra_joints': []}
    for arm in ('left', 'right'):
        joints = [f'{arm}_{i}' for i in range(7)]
        chain = ''
        for i, name in enumerate(joints):
            axis = ['1 0 0', '0 1 0', '0 0 1'][i % 3]
            chain += f'<body pos="0 0 .1"><joint name="{name}" axis="{axis}" range="-2 2"/><geom type="sphere" size=".02" mass=".1"/>'
            actuators.append(f'<velocity name="{name}" joint="{name}" kv="50" ctrlrange="-.4 .4"/>')
        chain += f'<body name="{arm}_tcp" pos=".1 0 0"/></body>' + '</body>'*6
        bodies.append(chain)
        bodies.append(f'<body><joint name="{arm}_grip" type="slide" range="0 .8"/><geom type="sphere" size=".01" mass=".1"/></body>')
        actuators.append(f'<position name="{arm}_grip" joint="{arm}_grip" kp="10" ctrlrange="0 .8"/>')
        mapping[arm] = {'joints': joints, 'actuators': joints, 'tcp_body': f'{arm}_tcp',
                        'gripper_actuator': f'{arm}_grip', 'gripper_closed_ctrl': .8, 'gripper_open_ctrl': 0.}
    path = tmp_path/'robot.xml'
    path.write_text('<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>'+''.join(bodies)+
                    '</worldbody><actuator>'+''.join(actuators)+'</actuator></mujoco>')
    return path, mapping


def test_kinematic_jacobian_matches_measured_pose_finite_difference(tmp_path):
    path, mapping = fixture(tmp_path)
    kinematics = RobotKinematics(mujoco.MjModel.from_xml_path(str(path)), mapping)
    q = np.linspace(-.2, .2, 7)
    reference = kinematics.observe(q, q)
    epsilon = 1e-6
    for arm in ('left', 'right'):
        columns = []
        for joint in range(7):
            plus, minus = q.copy(), q.copy()
            plus[joint] += epsilon; minus[joint] -= epsilon
            p = kinematics.observe(plus if arm == 'left' else q, plus if arm == 'right' else q)[arm]
            m = kinematics.observe(minus if arm == 'left' else q, minus if arm == 'right' else q)[arm]
            columns.append(pose_error(Pose(m['position'], m['quaternion_wxyz']),
                                      Pose(p['position'], p['quaternion_wxyz']))/(2*epsilon))
        np.testing.assert_allclose(np.stack(columns, axis=1), reference[arm]['jacobian'], atol=1e-6)
    mapping['extra_joints'] = ['measured_spine']
    with pytest.raises(ValueError, match='every calibrated'):
        kinematics.observe(q, q)


def test_probe_rejects_invalid_initial_state_and_unreachable_joint_target(tmp_path):
    path, mapping = fixture(tmp_path)
    with pytest.raises(ValueError, match='outside model limits'):
        ActuatorDiagnostic(path, mapping, [3.]*14)
    probe = ActuatorDiagnostic(path, mapping)
    command = np.zeros(23); command[0] = 3.
    with pytest.raises(ValueError, match='exceeds'):
        probe.step(command)
