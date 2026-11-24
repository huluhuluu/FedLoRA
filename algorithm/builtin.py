"""Built-in registrations. Importing plugins adds entries to the same registry."""

from algorithm.registry import register
from algorithm.sampling import IID, Shard, Dirichlet
from algorithm.grouping import NoneGrouping, Predefined, ResourceProfile, Distribution
from algorithm.selection import Uniform, GroupBalanced
from algorithm.initialization import Kaiming, Gaussian
from algorithm.Fed import FedAvg, SVD

for kind, names in {
    "partition": {"iid": IID, "shard": Shard, "dirichlet": Dirichlet},
    "grouping": {
        "none": NoneGrouping,
        "predefined": Predefined,
        "resource_profile": ResourceProfile,
        "distribution": Distribution,
    },
    "selection": {"uniform": Uniform, "group_balanced": GroupBalanced},
    "initialization": {"kaiming": Kaiming, "gaussian": Gaussian},
    "aggregation": {"fedavg": FedAvg, "svd": SVD},
}.items():
    for name, cls in names.items():
        register(kind, name)(cls)
