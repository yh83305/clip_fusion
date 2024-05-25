import cv2


class ImageComparator:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.old_orb = None
        self.new_orb = None

    def compute_ORB_descriptor(self, image):
        # 检测关键点和计算描述符
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        return descriptors

    def compute_similarity(self, descriptor1, descriptor2):
        # 使用 BFMatcher 进行匹配
        matches = self.bf.match(descriptor1, descriptor2)

        # 计算匹配点的相似度
        if len(matches) != 0:
            dis = sum(match.distance for match in matches) / len(matches)
            return dis
        else:
            return 100

    def judge_similarity(self, image, threshold):
        if self.old_orb is not None:
            self.new_orb = self.compute_ORB_descriptor(image)
            dis = self.compute_similarity(self.new_orb, self.old_orb)
            if dis > threshold:
                self.old_orb = self.new_orb
                return True
            else:
                return False
        else:
            self.old_orb = self.compute_ORB_descriptor(image)
            return True

