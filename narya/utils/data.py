from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from lxml import etree


def _parse_xml_file_keypoints(xml_file):
    """Get the keypoints and their id from a .xml file.

    Arguments:
        xml_file: String, the path to the .xml file
    Returns:
        keypoints: Dict, mapping each keypoints_id to their (x,y) location
    Raises:
        
    """
    keypoints = {}
    tree = etree.parse(xml_file)
    for i in range(len(tree.xpath("object/name"))):
        id_kp = int(tree.xpath("object/name")[i].text)
        x_kp = int(tree.xpath("object/keypoints/x1")[i].text)
        y_kp = int(tree.xpath("object/keypoints/y1")[i].text)

        keypoints[id_kp] = (y_kp, x_kp)

    return keypoints


def load_dump(dump_file):
    dump = []
    with open(dump_file, "rb") as in_fd:
        while True:
            try:
                step = six.moves.cPickle.load(in_fd)
            except EOFError:
                return dump
            dump.append(step)
    return dump
