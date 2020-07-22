import xml.etree.ElementTree as ET


class XML_Reader():
    def __init__(self, ):
        self.needles = []
        self.fishes = []
        self.file_path = ''

    def load_file(self):
        self.tree = ET.parse(self.file_path)
        self.root = self.tree.getroot()

    def list_objects(self):
        self.needles = []
        self.fishes = []
        for object in self.root.findall('object'):
            name = object.find('name').text
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            if name == 'needle':
                self.needles.append([xmin, ymin, xmax, ymax])
            else:
                self.fishes.append([xmin, ymin, xmax, ymax])


if __name__ == '__main__':
    xml_reader = XML_Reader()
    xml_reader.file_path = 'body0.xml'
    xml_reader.load_file()
    xml_reader.list_objects()
    print(xml_reader.needles, xml_reader.fishes)