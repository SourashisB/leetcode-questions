def numEquivDominoPairs(self, dominoes):
        """
        :type dominoes: List[List[int]]
        :rtype: int
        """
        # Count the occurrences of each domino
        count = {}
        for a, b in dominoes:
            # Sort the pair to ensure (a, b) and (b, a) are treated the same
            pair = (min(a, b), max(a, b))
            if pair in count:
                count[pair] += 1
            else:
                count[pair] = 1

        # Calculate the number of equivalent pairs
        result = 0
        for c in count.values():
            result += c * (c - 1) // 2

        return result