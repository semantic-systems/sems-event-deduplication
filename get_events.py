import time
from data.subclasses_of_natural_disaster import subclasses_of_natural_disaster
from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

P_INSTANCE_OF = "P31"
Q_NATURAL_DISASTERS = subclasses_of_natural_disaster
P_LOCATION = {"P131": "located in the administrative territorial entity",
              "P276": "location",
              "P1382": "partially coincident with",
              "P8138": "located in statistical territorial entity",
              "P706": "located in/on physical feature",
              "P4552": "mountain range",
              "P7153": "significant place",
              "P17": "country",
              "P625": "coordinate location",
              }
P_TIME = {"P585": "point in time",
          #"P4241": "refine date", # qualifier
          "P580": "start time",
          "P582": "end time",
          "P523": "temporal range start",
          "P524": "temporal range end",
          "P3415": "start period",
          "P3416": "end period",
          #"P2047": "duration"
          }

P_NUMBER_OF_DEATH = "P1120"
P_NUMBER_OF_INJURED = "P1339"
P_NUMBER_OF_MISSING = "P1446"
P_NUMBER_OF_CAUSALITIES = "P1590"
P_VICTIM = "P8032"


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