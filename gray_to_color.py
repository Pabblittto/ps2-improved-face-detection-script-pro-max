import sys


def grayToColor(gray: tuple):
    result = []

    for row in gray:
        newRow = []
        for element in row:
            newRow.append([element, element, element])
        result.append(newRow)

    return result


sys.modules[grayToColor] = grayToColor
