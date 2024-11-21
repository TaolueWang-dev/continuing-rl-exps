import unittest
from queue import PriorityQueue
from priorityqueue import PriorityQueueTop25Percent


class TestPriorityQueueTop25Percent(unittest.TestCase):

    def setUp(self):
        """
        每个测试用例执行前的初始化方法
        """
        self.pq_top_25 = PriorityQueueTop25Percent()

    def test_initialization(self):
        """
        测试初始化，确保队列为空
        """
        self.assertEqual(self.pq_top_25.pq_min.qsize(), 0)
        self.assertEqual(len(self.pq_top_25.pq_min.queue), 0)

    def test_push_one_element(self):
        """
        测试插入一个元素后，队列应当包含该元素
        """
        self.pq_top_25.push(5)
        self.assertEqual(self.pq_top_25.pq_min.qsize(), 1)
        self.assertEqual(self.pq_top_25.pq_min.queue[0], 5)

    def test_push_multiple_elements(self):
        """
        测试插入多个元素，并确保优先队列包含前 25% 大的元素
        """
        elements = [1, 3, 5, 7, 9, 2, 6, 8, 4, 10]
        for elem in elements:
            self.pq_top_25.push(elem)

        # 确保队列中包含前 25% 的元素（大约是3个元素）
        top_25_elements_length = max(1, int(len(elements) * 0.25))
        top_25_elements = sorted(self.pq_top_25.pq_min.queue, reverse=True)[:top_25_elements_length]  # 取前 25% 的大元素
        self.assertEqual(len(self.pq_top_25.pq_min.queue), top_25_elements_length)
        self.assertEqual(sorted(self.pq_top_25.pq_min.queue, reverse=True), top_25_elements)

    def test_get_avg(self):
        """
        测试 `get_avg()` 方法，确保返回正确的平均值
        """
        elements = [1, 3, 5, 7, 9, 2, 6, 8, 4, 10]
        for elem in elements:
            self.pq_top_25.push(elem)

        top_25_elements_length = max(1, int(len(elements) * 0.25))
        top_25_elements = sorted(self.pq_top_25.pq_min.queue, reverse=True)[:top_25_elements_length]

        # 计算前 25% 大的元素，平均值应该是 (40 + 50) / 2 = 45
        avg = self.pq_top_25.get_top25_avg()
        self.assertEqual(avg, sum(top_25_elements) / len(top_25_elements))

    def test_push_elements_in_order(self):
        """
        测试按升序插入元素，并验证队列中是否始终是前 25% 大的元素
        """
        elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for elem in elements:
            self.pq_top_25.push(elem)

        # 前 25% 的元素应该是最大的 2 个元素，即 [9, 10]
        self.assertEqual(sorted(self.pq_top_25.pq_min.queue, reverse=True), [10, 9])

    def test_push_elements_in_reverse_order(self):
        """
        测试按降序插入元素，并验证队列中是否始终是前 25% 大的元素
        """
        elements = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        for elem in elements:
            self.pq_top_25.push(elem)

        # 前 25% 的元素应该是最大的 2 个元素，即 [10, 9]
        self.assertEqual(sorted(self.pq_top_25.pq_min.queue, reverse=True), [10, 9])

    def test_edge_case_empty(self):
        """
        测试空队列的情况，确保计算平均值时不出错
        """
        avg = self.pq_top_25.get_top25_avg()
        self.assertEqual(avg, 0)  # 如果队列为空，平均值应该是0

    def test_edge_case_one_element(self):
        """
        测试只有一个元素时，确保返回的平均值是该元素
        """
        self.pq_top_25.push(100)
        avg = self.pq_top_25.get_top25_avg()
        self.assertEqual(avg, 100)


if __name__ == '__main__':
    unittest.main()
