for i in range(1, 101):
    continue
    print('<p id="q' + str(i) + '" style="display:none"> </strong>${original_text' + str(i) + '} <br></br></p>')

for i in range(1, 101):
    # print('<tr id="tr' + str(i) + '" style="display:none">\n<p id="q' + str(
    #     i) + '" style="display:none"> </strong>${original_text' + str(
    #     i) + '} <br></br> </p>\n<td>doesn’t bother</td>\n<td><crowd-slider id="slider' + str(
    #     i) + '" onchange="enableNext();" style="display:none; width: 800px; height: 20px;" name="mistakeRate' + str(
    #     i) + '" min="0" max="100"  required pin> </crowd-slider></td>\n<td></td><td></td><td></td><td></td><td>really bothers</td></tr>\n')

    print('<table id="table' + str(i) + '" style="display:none">\n<tr colspan="3"><p id="q' + str(
        i) + '" style="display:none"> </strong>${original_text' + str(
        i) + '} <br /> </p></tr>\n<tr><td>doesn’t bother</td>\n<td><crowd-slider id="slider' + str(
        i) + '" onchange="enableNext();" style="display:none; width: 800px; height: 20px;" name="mistakeRate' + str(
        i) + '" min="0" max="100" required pin> </crowd-slider></td>\n<td></td><td></td><td></td><td></td><td>really bothers</td>\n</tr></table>\n')

    #
    # print('<table>\n<tr colspan="3"><p id="q' + str(
    #     i) + '" style="display:none"> </strong>${original_text' + str(
    #     i) + '} <br /> </p></tr>\n<tr id="tr' + str(
    #     i) + '" style="display:none">\n<td>doesn’t bother</td>\n<td><crowd-slider id="slider' + str(
    #     i) + '" onchange="enableNext();" style="display:none; width: 800px; height: 20px;" name="mistakeRate' + str(
    #     i) + '" min="0" max="100"  required pin> </crowd-slider></td>\n<td></td><td></td><td></td><td></td><td>really bothers</td>\n</tr>\n</table>\n')
