import unittest

from src.knn import minkowski


class TestMimkwoski(unittest.TestCase):

    def samelenght(self):
        a = [2,4]
        b = [3,5]
        self.assertEqual(minkowski(a,b), 4)


def main():
    test = TestMimkwoski()
    test.samelenght()
if __name__ == '__main__':
    unittest.main()