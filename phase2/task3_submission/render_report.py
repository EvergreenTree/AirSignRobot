"""Render the concise companion report; requires reportlab (authoring only)."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / 'evidence/summary.json').read_text())
NAVY = colors.HexColor('#173442')
TEAL = colors.HexColor('#007E87')
GRAY = colors.HexColor('#53636D')
PALE = colors.HexColor('#EEF4F5')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('TitleCustom', fontName='Helvetica-Bold', fontSize=27, leading=31, textColor=NAVY, spaceAfter=12))
styles.add(ParagraphStyle('SubCustom', fontName='Helvetica', fontSize=11, leading=15, textColor=GRAY, spaceAfter=14))
styles.add(ParagraphStyle('HCustom', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=TEAL, spaceBefore=12, spaceAfter=7))
styles.add(ParagraphStyle('BCustom', fontName='Helvetica', fontSize=10.2, leading=14.5, spaceAfter=8, textColor=NAVY))
styles.add(ParagraphStyle('SmallCustom', fontName='Helvetica', fontSize=8.6, leading=11.5, textColor=GRAY, spaceAfter=6))
styles.add(ParagraphStyle('CellCustom', fontName='Helvetica', fontSize=9, leading=12, textColor=NAVY))
styles.add(ParagraphStyle('CodeCustom', fontName='Courier', fontSize=8, leading=11, textColor=NAVY, backColor=PALE, borderPadding=7, spaceAfter=10))
story = []

def p(text, style='BCustom'):
    return Paragraph(text, styles[style])

def add(text, style='BCustom'):
    story.append(p(text, style))

def table(rows, widths):
    data = [[p(str(cell), 'CellCustom') for cell in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PALE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, 0), .7, TEAL),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#CBD7DB')),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 7))

add('AirSign', 'SubCustom')
add('Task 3 Phase II<br/>Technical progress report', 'TitleCustom')
add('EBiM Assisted Living &amp; Feeding | September 11-12, 2026<br/>Supervised real-hardware work on Mobile FR3 Duo', 'SubCustom')
add('<b>Outcome.</b> We established measured-state access, wrist-camera mapping and bounded physical motion. The right arm approached the plate, but contact, grasp and lift were not verified. No completed Task 3 stage or benchmark score is claimed.')
table([
    ['<b>Measured progress</b>', '<b>Evidence boundary</b>'],
    [f"<b>{summary['right_arm']['net_displacement_m']*100:.2f} cm</b> net right-arm displacement", 'Difference between retained initial and final O_T_EE translations; not path length or plate motion.'],
    [f"<b>{summary['base']['right_turn_degrees_from_step2_reference']:.2f} degrees</b> estimated right turn", 'Sum of settled independent scan-registration increments, steps 2-21; not completed autonomous navigation.'],
    ['Five finite right-arm calls', 'The final call stdout was not retained. The final measured sample reports Idle and no current errors.'],
], [180, 330])
add('Visual evidence', 'HCustom')
imgs = [Image(str(HERE / 'evidence/images' / name), width=248, height=186) for name in ('before-approach.png', 'before-final-step.png')]
photos = Table([imgs, [p('Before the first arm translation.', 'SmallCustom'), p('After four translations, before the fifth.', 'SmallCustom')]], colWidths=[255, 255])
photos.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),7)]))
story.append(photos)
add('Selected original right-wrist RGB images. No post-fifth-step image is archived; the endpoint is supported by telemetry. The plate remained on the table.', 'SmallCustom')

story.append(PageBreak())
add('Method and operating boundary', 'TitleCustom')
add('Working primitives, a feedback program, and the integration still required', 'SubCustom')
table([
    ['<b>Component</b>', '<b>Implemented / observed</b>'],
    ['Native arm control', 'Finite pose-anchored Cartesian translation on the RT host. Existing orientation retained; Cartesian impedance; no homing, automatic recovery or grasp command.'],
    ['Base control', 'Finite clockwise steps with fresh front/rear scan registration. SSH carried supervisory requests, not servo samples.'],
    ['Visual supervision', 'Wrist RGB-D, measured arm poses and nominal vendor geometry; image regions and waypoints selected by a remote agent.'],
    ['Task 3 program', 'JSONL observation-to-action state machine with navigation, grasp, transport, feeding and cleanup stages. Software tested; not connected end-to-end in this trial.'],
], [124, 386])
add('Measured-state control', 'HCustom')
add('The native arm helper enforces real-time scheduling, initial idle/error checks and agreement with the reviewed measured pose. Requests are limited to 50 mm and 3-8 seconds. During motion it monitors forces/torques, collision flags, joint speed/change, orientation change, travel and callback timing. These are engineering guards, not certified collision protection or calibrated grasp forces.')
add('Normal control ownership and FCI activation are handled in an interactive session; owned control is released on exit. Credentials are entered at a prompt and are absent from the package. Hardware readiness must be re-established for a future attended run.')
add('Calibration findings', 'HCustom')
add('Two D405 wrists supplied 640x480 RGB and raw millimetre depth. Current camera labels were resolved with kinematics and cross-camera features. An identity TF between wrists was a placeholder. RGB and depth had different intrinsics, so registration required explicit treatment.')
add('The measured flange-to-configured-EE offset was 174 mm. Nominal vendor mounts supported geometry estimates, but full hand-eye and finger-contact calibration were incomplete. Plate coordinates were relative to nominal spine zero. The existing tilted tool orientation was retained; thin-rim grasping and bimanual load sharing were not validated.')
add('Source preservation', 'HCustom')
add('Seven final on-site source files are archived byte-for-byte, alongside read-only probes, a new CMake wrapper, a reusable skill and raw evidence. The sources were compiled and used on site; the packaging wrapper was added afterward and has not been rebuilt on the robot. Vendor libraries and models remain site dependencies.')

story.append(PageBreak())
add('Validation and declarations', 'TitleCustom')
add('Reproducible review is separate from autonomous physical task execution', 'SubCustom')
table([
    ['<b>Check</b>', '<b>Result / limit</b>'],
    ['Evidence review', 'Checks SHA-256 hashes and recomputes arm displacement and cumulative scan-registration yaw from retained records.'],
    ['Software validation', '31 existing control/scoring tests plus 3 synthetic JSONL cases for stale, missing and malformed observations. See validation/local-review.txt for execution output.'],
    ['Docker', 'CPU-only review and JSONL interface provided. Build/run not executed in the available environment after departure; remains unverified.'],
    ['Physical Task 3', 'Partial supervised approach only. No plate contact, grasp, lift, feeding or full stage completion is asserted.'],
], [124, 386])
add('Local review entrypoint', 'HCustom')
add('python phase2/task3_submission/review.py review<br/>python phase2/task3_submission/review.py self-test<br/>python phase2/task3_submission/review.py describe', 'CodeCustom')
add('Install requirements-review.txt in an isolated Python 3.12 environment first. The dedicated Dockerfile is phase2/Dockerfile.task3. Review needs no GPU, ROS, credentials or runtime Internet. Full commands, environment details and the observation contract are linked in README.md.', 'SmallCustom')
add('Human assistance and data', 'HCustom')
add('The operator initialized hardware, supervised the emergency stop, supplied high-level directions and moved the desk in front of the robot. A remote visual agent reviewed finite steps. No left-arm trajectory was executed for the plate approach. The current object-pose declaration is <b>Partly - see Notes</b>; autonomous end-to-end object perception is not supplied.')
add('No Task 3 trajectory data was used for training and no Task 3 learned weights are claimed. No simulator ground-truth poses drove the real-hardware steps. Task 1/2 results elsewhere in the repository are separate.')
add('Remaining integration and submission', 'HCustom')
add('A deployable policy still needs the live perception/transport bridge, verified grasp and contact calibration, and complete task-stage trials. An agent-dependent route also needs an explicit launcher, model/tools, credential and network requirements, and organizer acceptance; the skill alone is not a runnable policy.')
add('The current Phase II form asks for a public exact commit and working launch commands; the team advancement email controls the deadline. The submissions README advertises a technical-report route, but the current chooser exposes only the policy form. Report-route acceptance, registration details and publication remain pending.')
add('<link href="https://github.com/EBiM-Benchmark/submissions/blob/main/.github/ISSUE_TEMPLATE/phase2-submission.yml" color="#007E87">Official Phase II form</link> | <link href="https://github.com/EBiM-Benchmark/submissions" color="#007E87">Submission repository</link><br/>Full technical narrative: REPORT.md. Machine-readable result: evidence/summary.json. Issue draft: ISSUE_DRAFT.md.', 'SmallCustom')

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#CBD7DB'))
    canvas.line(42, 38, A4[0]-42, 38)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(42, 25, 'AirSign | Task 3 Phase II | Technical progress')
    canvas.drawRightString(A4[0]-42, 25, str(doc.page))

doc = SimpleDocTemplate(str(HERE / 'AirSign-Task3-PhaseII-Report.pdf'), pagesize=A4,
                        rightMargin=42, leftMargin=42, topMargin=40, bottomMargin=53,
                        title='AirSign Task 3 Phase II - Technical Progress', author='AirSign')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(HERE / 'AirSign-Task3-PhaseII-Report.pdf')
