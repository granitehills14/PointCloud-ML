import json
import pdal 

def neighborclassifier(neighborhood, candidate, point_cloud):
    
    k = neighborhood["k"]
    unclass_labels = neighborhood["unclassified_labels"]
    arrays=[candidate]

    domain = ", ".join(
    f"Classification[{lbl}:{lbl}]"
    for lbl in unclass_labels
    )

    pipeline_def = json.dumps({
        "pipeline": [
            {"type" : "readers.numpy"},
            {
                "type" : "filters.neighborclassifier",
                "candidate" : arrays[0],
                "k" : k,
                "domain" : domain
            }
        ]
    })

    pipeline = pdal.Pipeline(pipeline_def, arrays.append([point_cloud]))
    pipeline.execute()

    classified_pc = pipeline.arrays[1]

    return classified_pc


