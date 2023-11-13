import time
from data.subclasses_of_natural_disaster import subclasses_of_natural_disaster
from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

P_INSTANCE_OF = "P31"
Q_NATURAL_DISASTERS = subclasses_of_natural_disaster


def instance_of_natural_disaster(item: WikidataItem) -> bool:
    claim_group = item.get_claim_group(P_INSTANCE_OF)
    disaster_qids = [
        claim.mainsnak.datavalue.value["id"]
        for claim in claim_group
        if claim.mainsnak.snaktype == "value"
    ]
    return bool(list(set(Q_NATURAL_DISASTERS).intersection(set(disaster_qids))))


# create an instance of WikidataJsonDump
wjd_dump_path = "/data2/huang/latest-all.json.gz"
wjd = WikidataJsonDump(wjd_dump_path)


# create an iterable of WikidataItem representing politicians
natural_disaster = []
t1 = time.time()
for ii, entity_dict in enumerate(wjd):
    if entity_dict["type"] == "item":
        entity = WikidataItem(entity_dict)
        if instance_of_natural_disaster(entity):
            natural_disaster.append(entity)

    if ii % 1000 == 0:
        t2 = time.time()
        dt = t2 - t1
        print(
            "found {} natural disasters among {} entities [entities/s: {:.2f}]".format(
                len(natural_disaster), ii, ii / dt
            )
        )
        if natural_disaster:
            print(natural_disaster)


# write the iterable of WikidataItem to disk as JSON
out_fname = "./filtered_natural_disaster_entities_included_subclasses.json"
dump_entities_to_json(natural_disaster, out_fname)
wjd_filtered = WikidataJsonDump(out_fname)

# # load filtered entities and create instances of WikidataItem
# for ii, entity_dict in enumerate(wjd_filtered):
#     item = WikidataItem(entity_dict)