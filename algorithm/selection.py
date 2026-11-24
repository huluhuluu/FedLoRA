"""Sample logical clients without coupling to worker IDs."""


class Uniform:
    def __init__(self, clients_per_round=1):
        if clients_per_round < 1:
            raise ValueError("clients_per_round must be positive")
        self.count = clients_per_round

    def select(self, clients, groups, context):
        ids = sorted(clients)
        return sorted(
            context.rng.choice(ids, min(self.count, len(ids)), replace=False).tolist()
        )


class GroupBalanced(Uniform):
    def select(self, clients, groups, context):
        queues = []
        for name in sorted(groups):
            available = [cid for cid in groups[name] if cid in clients]
            if available:
                queues.append(context.rng.permutation(available).tolist())
        # Small quotas must not permanently favor the first groups.
        queues = [queues[index] for index in context.rng.permutation(len(queues))]
        selected = []
        while queues and len(selected) < min(self.count, len(clients)):
            remaining = []
            for queue in queues:
                if len(selected) < self.count:
                    selected.append(queue.pop())
                if queue:
                    remaining.append(queue)
            queues = remaining
        return sorted(selected)
