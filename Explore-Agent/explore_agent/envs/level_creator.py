import matplotlib.pyplot as plt
import numpy as np
from exploring_gym import line_intersect

# ─── README ─────────────────────────────────────────────────────────────────────
# 1) execute this script and click in the window to create points
# 2) created points get printed in the console
# 3) copy points from console into the script. Render them for reference if needed 
# 4) repeat with 1
# 5) once you have all your points, bring them in the format present in 
#    exploring_gym.py (np.array with respective shape)
# ────────────────────────────────────────────────────────────────────────────────

def draw_line():
    ax = plt.gca()
    xy = plt.ginput(10)
    x = [p[0] for p in xy]
    y = [p[1] for p in xy]
    line = plt.plot(x, y)
    ax.figure.canvas.draw()
    print(xy)
    
fig = plt.figure(figsize=(16, 10))
plt.xlim(0, 1600)
plt.ylim(0, 1000)

l1 = np.array([(307.09677419354836, 972.3603050256356), (428.3870967741935, 973.6616089321499), (554.8387096774193, 967.1550893995782), (660.6451612903224, 960.6485698670067), (732.9032258064517, 941.1290112692918), (767.741935483871, 904.6925018868906), (772.9032258064517, 839.6273065611742), (770.3225806451612, 768.0555917028862), (766.4516129032259, 702.9903963771699),(725.1612903225807, 628.8160737058533), (686.4516129032259, 568.9560940061942), (646.4516129032259, 529.9169768107643), (601.2903225806451, 496.0830752413919), (539.3548387096773, 463.5504775785336), (495.4838709677419, 421.90875257007525), (485.16129032258067, 386.77354709418836), (501.93548387096774, 362.04877287041614), (563.8709677419354, 350.3370377117872), (621.9354838709678, 368.5552924029878),(699.3548387096773, 414.1009291309892), (752.258064516129, 476.5635166436769), (790.9677419354839, 566.3534861931655), (806.4516129032259, 656.1434557426542), (805.1612903225807, 731.6190823204852), (797.4193548387098, 820.1077479634594), (806.4516129032259, 917.7055409520337), (860.6451612903227, 976.2642167451786), (980.6451612903227, 985.3733440907788),(1100.6451612903227, 985.3733440907788), (1291.6129032258066, 982.7707362777501), (1496.774193548387, 948.9368347083778), (1571.6129032258063, 804.4921010852875), (1560.0, 624.9121619863101), (1436.1290322580646, 546.8339275954506), (1278.7096774193549, 546.8339275954506),(1200.0, 540.3274080628789), (1179.3548387096776, 505.1922025869921), (1188.3870967741937, 470.0569971111053), (1214.1935483870968, 449.2361346068761), (1314.8387096774193, 442.7296150743044),(1390.967741935484, 433.6204877287042), (1531.6129032258063, 332.1187830205866), (1563.8709677419356, 209.79621580823988), (1548.3870967741937, 101.7879915675507), (1366.4516129032259, 12.61366889623403), (1067.0967741935483, 8.914972802748366), (738.0645161290322, 5.094110298519126), (383.22580645161287, 11.998022018062102), (145.80645161290323, 21.107149363662415),(77.4193548387097, 41.92801186789163), (12.903225806451587, 97.8840798480077),(6.451612903225794, 224.11055877989747), (11.612903225806463, 411.4983213179606), (14.193548387096769, 585.8730447908804), (21.935483870967744, 805.7934049918017),(61.935483870967744, 896.8846784478046), (144.51612903225805, 948.9368347083778),(307.09677419354836, 972.3603050256356)])
n = len(l1)
line_combined1 = np.zeros((n-1, 4))
line_combined1[:,[0,1]] = l1[:n-1,[0,1]]
line_combined1[:,[2,3]] = l1[1:n,[0,1]]
for i, line in enumerate(line_combined1):
    x = [line[0], line[2]]
    y = [line[1], line[3]]
    plt.plot(x,y,)

l2 = np.array([(198.70967741935482, 424.51136038310386), (194.83870967741933, 565.0521822866513), (233.5483870967742, 700.3877885641411), (326.4516129032258, 766.754287796372), (419.3548387096774, 760.2477682638003), (478.7096774193548, 708.1956120032271), (470.96774193548384, 606.6939072951097), (427.09677419354836, 529.9169768107643),(303.22580645161287, 391.9787627202457), (254.19354838709677, 296.98357754469976), (263.22580645161287, 233.21968612549773), (329.03225806451616, 179.86622595841033), (438.7096774193548, 162.94927517372406), (554.8387096774193, 165.55188298675273), (667.0967741935483, 175.96231423886735), (810.3225806451612, 226.7131665929261), (907.0967741935485, 304.79140098378576), (975.483870967742, 421.90875257007525),(987.0967741935483, 477.86482055019127), (990.9677419354839, 581.9691330713374), (978.0645161290322, 676.9643182468833), (967.741935483871, 770.6581995159149), (1025.8064516129032, 825.3129635895166), (1141.9354838709678, 829.2168753090596), (1301.9354838709678, 818.8064440569451), (1385.8064516129032, 783.6712385810581),(1367.741935483871, 747.2347291986571), (1326.4516129032259, 722.5099549748847),(1198.7096774193549, 689.9773573120267),(1114.8387096774193, 675.663014340369), (1036.1290322580646, 623.6108580797959), (1015.483870967742, 545.5326236889364), (1018.0645161290322, 449.2361346068761), (1023.2258064516129, 338.62530255315824), (1095.483870967742, 277.4640189469849), (1300.6451612903227, 254.04054862972697), (1401.2903225806451, 252.73924472321266), (1418.0645161290322, 234.5209900320121),(1403.8709677419356, 215.00143143429716), (1356.1290322580646, 185.07144158446766), (1269.6774193548388, 183.7701376779533), (1056.774193548387, 198.0844806496109), (898.0645161290322, 188.97535330401064),(658.0645161290322, 140.01928532389456), (576.7741935483871, 134.32058923040887), (437.4193548387097, 134.32058923040887), (348.3870967741935, 136.92319704343754), (261.93548387096774, 169.4557947062957), (238.70967741935482, 208.49491190172552), (209.0322580645161, 332.1187830205866),(198.70967741935482, 424.51136038310386)])
n = len(l2)
line_combined2 = np.zeros((n-1, 4))
line_combined2[:,[0,1]] = l2[:n-1,[0,1]]
line_combined2[:,[2,3]] = l2[1:n,[0,1]]
for i, line in enumerate(line_combined2):
    x = [line[0], line[2]]
    y = [line[1], line[3]]
    plt.plot(x,y,)

# calculate intersection between level and goal
goals = np.array([[ 386.,  743.,  444.,  987.], [ 437.,  710.,  769.,  934.], [ 457.,  677.,  730.,  629.], [ 440.,  595.,  594.,  471.], [ 345.,  478.,  510.,  404.], [ 299.,  184.,  503.,  373.], [ 600.,  155.,  585.,  369.], [ 834.,  227.,  701.,  431.], [ 999.,  514.,  781.,  558.], [ 988.,  732.,  794.,  732.], [1022.,  797.,  840.,  973.], [1107.,  821., 1071.,  996.], [1324.,  801., 1397.,  986.], [1356.,  768., 1587.,  775.], [1341.,  756., 1514.,  564.], [1305.,  534., 1258.,  709.], [1204.,  517., 1072.,  662.], [1191.,  486., 1005.,  474.], [1235.,  458., 1169.,  252.], [1374.,  242., 1440.,  427.], [1400.,  230., 1568.,  173.], [1338.,  197., 1405.,   17.], [1111.,  201., 1133.,    3.], [ 836.,  187.,  862.,    1.], [ 617.,  148.,  627.,    3.], [ 439.,  143.,  419.,    5.], [ 275.,  174.,  217.,    9.], [ 235.,  257.,    3.,  129.], [ 213.,  386.,    4.,  392.], [ 221.,  592.,    8.,  649.], [ 253.,  673.,    1.,  784.], [ 315.,  735.,  215.,  977.]])
intersect_x = [] # for scatter
intersect_y = [] # for scatter
goals_new = [] # for export
for goal in goals:
    goal_new = [1] * 4
    for line in line_combined1:
        result = line_intersect(*goal, *line)
        if result:
            intersect_x.append(result[0])
            intersect_y.append(result[1])
            goal_new[0] = result[0]
            goal_new[1] = result[1]
    for line in line_combined2:
        result = line_intersect(*goal, *line)
        if result:
            intersect_x.append(result[0])
            intersect_y.append(result[1])
            goal_new[2] = result[0]
            goal_new[3] = result[1]
    goals_new.append(goal_new)

plt.scatter(intersect_x, intersect_y)
goals_new = np.round(np.array(goals_new))
print(goals_new)

# to use goals_new in the environment, manually reformat as follows:
goals_reformated = np.array([[441.,  973.,  391.,  762.], [751.,  922.,  459.,  725.], [726.,  630.,  476.,  674.], [578.,  484.,  457.,  582.], [493.,  412.,  370.,  467.], [498.,  368.,  311.,  195.], [586.,  357.,  599.,  170.], [707.,  423.,  826.,  239.], [787.,  557.,  989.,  516.], [805.,  732.,  972.,  732.], [849.,  964., 1009.,  810.], [1073.,  985., 1106.,  828.], [1389.,  967., 1327.,  808.], [1570.,  774., 1378.,  769.], [1496.,  584., 1356.,  740.], [
                            1302.,  547., 1259.,  705.], [1193.,  529., 1080.,  653.], [1184.,  486., 1017.,  475.], [1232.,  448., 1174.,  268.], [1432.,  404., 1378.,  253.], [1559.,  176., 1412.,  226.], [1400.,   29., 1343.,  185.], [1132.,   10., 1112.,  194.], [861.,    7.,  837.,  177.], [627.,    7.,  618.,  137.], [420.,   11.,  438.,  134.], [220.,   18.,  272.,  166.], [11.,  133.,  228.,  253.], [11.,  392.,  203.,  386.], [16.,  647.,  204.,  597.], [21.,  775.,  229.,  684.], [222.,  960.,  307.,  753.]])

for i, goal in enumerate(goals_reformated):
    x = [goal[0], goal[2]]
    y = [goal[1], goal[3]]
    plt.plot(x,y,label=str(i))

for _ in range(10):
    draw_line()
