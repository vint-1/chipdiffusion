import utils
import os
import pickle

def cluster_dataset(dataset_name, out_name, num_clusters, save_image = False, verbose = False):
    train_set, val_set = utils.load_graph_data(dataset_name)
    out_dir = os.path.join("datasets", "graph", out_name)
    try:
        os.makedirs(os.path.join(out_dir, "images") if save_image else out_dir)
    except FileExistsError:
        pass
    idx = 0
    print(f"clustering into {num_clusters}. saving to {out_dir}")
    # cluster each set
    for x, cond in (train_set + val_set):
        import ipdb; ipdb.set_trace()
        out_idx = cond.file_idx if "file_idx" in cond else idx
        if not (os.path.exists(os.path.join(out_dir, f"graph{out_idx}.pickle")) and os.path.exists(os.path.join(out_dir, f"output{out_idx}.pickle"))):
            cluster_and_save(out_dir, num_clusters, cond, x, save_image=save_image, output_id=idx, verbose=verbose)
        idx += 1
    print(f"finished {idx} samples")

def cluster_and_save(out_dir, num_clusters, cond, x, save_image = False, output_id = 0, verbose = False):
    """
    perform clustering of one sample, then save output
    """
    # cluster
    cluster_cond, cluster_x = utils.cluster(cond, num_clusters, placements=x.unsqueeze(dim=0), verbose=verbose)
    cluster_x = cluster_x.squeeze(dim=0) # (V, 2)
    # save image too
    out_idx = cond.file_idx if "file_idx" in cond else output_id
    if save_image:
        image = utils.visualize_placement(cluster_x, cluster_cond, plot_pins=True, plot_edges=False, img_size=(1024, 1024))
        utils.debug_plot_img(image, os.path.join(out_dir, "images", f"clustered{out_idx}"))
        image = utils.visualize_placement(x, cond, plot_pins=False, plot_edges=False, img_size=(1024, 1024))
        utils.debug_plot_img(image, os.path.join(out_dir, "images", f"original{out_idx}"))

    # postprocess
    postprocessed_x, postprocessed_cond = utils.postprocess_placement(cluster_x, cluster_cond, process_graph=True)
    postprocessed_x = postprocessed_x.cpu().numpy()

    # save outputs
    x_path = os.path.join(out_dir, f"output{out_idx}.pickle")
    with open(x_path, 'wb') as f:
        pickle.dump(postprocessed_x, f)
    
    cond_path = os.path.join(out_dir, f"graph{out_idx}.pickle")
    with open(cond_path, 'wb') as f:
        pickle.dump(postprocessed_cond, f)

if __name__=="__main__":

    dataset_name = ""
    num_clusters = 512
    save_image = True
    verbose = True

    cluster_dataset(dataset_name, f"{dataset_name}-cluster{num_clusters}", num_clusters, save_image=save_image, verbose=verbose)