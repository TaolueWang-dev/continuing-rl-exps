from queue import PriorityQueue
import heapq

class PriorityQueueTop25Percent():

    def __init__(self):
        self.pq_max = PriorityQueue()
        self.pq_min = PriorityQueue()
        self.elements = []
        self.top_25_percent_element_length = 0

    def push(self, element):
        self.elements.append(element)
        self.top_25_percent_element_length = max(1, int(len(self.elements) * 0.25))
        self.pq_max.put(-1 * element)
        if self.pq_min.qsize() < self.top_25_percent_element_length:
            self.pq_min.put(-1 * self.pq_max.get())
        else:
            if element > self.pq_min.queue[0]:
                self.pq_min.get()
                self.pq_min.put(element)

    def get_top25_avg(self):
        sum = 0
        for item in self.pq_min.queue:
            sum += item
        if self.pq_min.qsize() == 0:
            return 0
        else:
            return sum / self.pq_min.qsize()


if __name__ == '__main__':
    pq = PriorityQueueTop25Percent()
