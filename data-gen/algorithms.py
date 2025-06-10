from distributions import get_distribution
import utils
import torch
import shapely
import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils as tgu

class V1:
    def __init__(
            self, 
            max_instance, 
            stop_density,
            max_attempts_per_instance,
            aspect_ratio_dist, 
            instance_size_dist, 
            num_terminals_dist,
            edge_dist,
            source_terminal_dist,
        ):
        self.max_instance = max_instance
        self.stop_density = stop_density
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist
        self.num_terminals_dist = num_terminals_dist
        self.edge_dist = edge_dist
        self.max_attempts_per_instance = max_attempts_per_instance
        self.source_terminal_dist = source_terminal_dist

    def sample(self):
        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,))

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        # place samples individually
        placement = Placement()
        density = 0
        for (x_size, y_size) in zip(x_sizes, y_sizes):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([-1.0, -1.0]), "high": torch.tensor([1.0-x_size, 1.0-y_size])}
            candidate_dist =  get_distribution("uniform", dist_params)
            
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()
                # print(instance_idx, attempt_num, candidate_pos)
                if placement.check_legality(candidate_pos[0].item()+x_size/2, candidate_pos[1].item()+y_size/2, x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item()+x_size/2, candidate_pos[1].item()+y_size/2, x_size, y_size)
                    break
            density += (x_size * y_size)/4.0
            if density >= self.stop_density:
                break
        
        positions = placement.get_positions()
        sizes = placement.get_sizes()
        positions = positions - sizes/2 # use bottom left as coordinates
        num_instances = positions.shape[0]

        # sample number of terminals
        num_terminals = get_distribution(**self.num_terminals_dist).sample((num_instances,)).int() # TODO condition dist on area of instance per Rent's
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="bottom_left")

        # generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets # (V, T, 2)
        terminal_distances = self.get_terminal_distances(terminal_positions)
        edge_exists = get_distribution(**self.edge_dist).sample(terminal_distances) # (V, T, V, T)
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        # delete edges between same instance, among other things
        edge_exists = self.process_edge_matrix(edge_exists, is_source, num_terminals)

        # convert to edge list and generate attributes
        edge_index, edge_attr = self.generate_edge_list(edge_exists, terminal_offsets)
        mask = placement.get_mask()

        data = Data(x=sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
        return positions, data
    
    def get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals, reference="center"):
        # TODO check reference points for these
        # NOTE here we assume reference point (for computing offset) is center of instance
        # NOTE outputs use whatever units are being used for sizes
        # x_sizes: (num_instances)
        # y_sizes: (num_instances)
        # max_num_terminals: int
        half_perim = (x_sizes + y_sizes)
        
        terminal_locations = get_distribution("uniform", {"low": 0, "high": half_perim}).sample((max_num_terminals,)) # (max_term, num_instances)
        terminal_flip = get_distribution("bernoulli", {"probs": 0.5}).sample((max_num_terminals, x_sizes.shape[0])) # (max_term, num_instances)
        terminal_flip = (2 * terminal_flip) - 1

        x_sizes = x_sizes.unsqueeze(dim=0)
        y_sizes = y_sizes.unsqueeze(dim=0)
        terminal_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2) 
        terminal_offset_y = torch.clamp(terminal_locations-x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)

        terminal_offset_x = terminal_flip * terminal_offset_x
        terminal_offset_y = terminal_flip * terminal_offset_y

        if reference == "bottom_left":
            terminal_offset_x += x_sizes/2
            terminal_offset_y += y_sizes/2

        terminal_offset = torch.stack((terminal_offset_x, terminal_offset_y), dim=-1).movedim(1, 0) # (num_instances, max_term, 2)
        return terminal_offset

    def get_terminal_distances(self, terminal_positions, norm_order=1):
        # given global terminal positions (V, T, 2)
        # compute and return pairwise L1 distances between terminals
        # optionally, can specify x for Lx distance using norm_order
        V, T, _ = terminal_positions.shape
        t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
        t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
        delta_pos = t_pos_1 - t_pos_2 # (V, T, V, T, 2)
        distance = torch.norm(delta_pos, p=norm_order, dim=-1) # (V, T, V, T)
        return distance

    def process_edge_matrix(self, edge_exists, is_source, num_terminals):
        # edge_existence tensor (V, T, V, T)
        # is_source int tensor (V, T) (0=sink, 1=source)
        # num terminals int tensor (V)
        # process  as follows:
        # remove all edges ending in a source
        # remove all edges originating from a sink
        # remove all edges between same instance
        # remove all edges starting or ending in nonexistent terminal
        V, T, _, _ = edge_exists.shape
        assert is_source.shape == edge_exists.shape[:2]
        assert num_terminals.shape == (V,)

        # generate terminal filter
        terminal_filter = torch.zeros((V, T))
        for i, num_terminal in enumerate(num_terminals):
            terminal_filter[i, :num_terminal] = 1

        source_filter = (terminal_filter * is_source).view(V, T, 1, 1)
        sink_filter = (terminal_filter * (1-is_source)).view(1, 1, V, T)
        self_edge_filter = (1-torch.eye(V)).view(V, 1, V, 1)
        
        edges = edge_exists * source_filter
        edges = edges * sink_filter
        edges = edges * self_edge_filter
        return edges

    def generate_edge_list(self, edge_exists, terminal_offsets):
        V, T, _, _ = edge_exists.shape
        edges = torch.nonzero(edge_exists) # (E, 4:v,t,v,t) int64
        edge_index_forward = edges[:,(0,2)]
        edge_index_reverse = edges[:,(2,0)]

        edge_attr_source = terminal_offsets[edges[:,0], edges[:,1], :] # (E, 2)
        edge_attr_sink = terminal_offsets[edges[:,2], edges[:,3], :] # (E, 2)
        edge_attr_forward = torch.concat((edge_attr_source, edge_attr_sink), dim=-1)
        edge_attr_reverse = torch.concat((edge_attr_sink, edge_attr_source), dim=-1)
        
        # create undirected edge index and attr
        edge_index = torch.concat((edge_index_forward, edge_index_reverse), dim=0).T # (2, E)
        edge_attr = torch.concat((edge_attr_forward, edge_attr_reverse), dim=0) # (E, 4)
        
        # return copy
        return edge_index.clone(), edge_attr.clone()

class V2:
    def __init__(
            self, 
            max_instance, 
            stop_density_dist,
            max_attempts_per_instance,
            aspect_ratio_dist, 
            instance_size_dist, 
            num_terminals_dist,
            edge_dist,
            source_terminal_dist,
            interior_terminals_dist = None,
            interior_terminals_loc = "uniform",
            zero_edge_attr = False,
            distance_norm_order = 1, # order of norm for measuring distance
        ):
        self.max_instance = max_instance
        self.stop_density_dist = stop_density_dist
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist
        self.num_terminals_dist = num_terminals_dist
        self.interior_terminals_dist = interior_terminals_dist
        self.interior_terminals_loc = interior_terminals_loc
        self.edge_dist = edge_dist
        self.max_attempts_per_instance = max_attempts_per_instance
        self.source_terminal_dist = source_terminal_dist
        self.zero_edge_attr = zero_edge_attr
        self.distance_norm_order = distance_norm_order

    def sample(
            self, 
            size_dist_timer=None, # optional timers
            place_timer=None,
            terminal_timer=None,
            edge_timer=None,
        ):

        size_dist_timer.start() if size_dist_timer else None

        # Generate stop density
        stop_density = get_distribution(**self.stop_density_dist).sample()

        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,))

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        size_dist_timer.stop() if size_dist_timer else None
        place_timer.start() if place_timer else None

        # place samples individually
        placement = Placement()
        density = 0
        for (x_size, y_size) in zip(x_sizes, y_sizes):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([(x_size/2)-1.0, (y_size/2)-1.0]), "high": torch.tensor([1.0-(x_size/2), 1.0-(y_size/2)])}
            candidate_dist =  get_distribution("uniform", dist_params)
            
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()
                # print(instance_idx, attempt_num, candidate_pos)
                if placement.check_legality(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size)
                    break
            density += (x_size * y_size)/4.0
            if density >= stop_density:
                break
        
        positions = placement.get_positions()
        sizes = placement.get_sizes()
        num_instances = positions.shape[0]

        place_timer.stop() if place_timer else None
        terminal_timer.start() if terminal_timer else None

        # sample number of terminals
        instance_area = sizes[:, 0] * sizes[:, 1]
        num_terminals = get_distribution(**self.num_terminals_dist).sample(instance_area).int()
        # num_terminals = torch.clip(num_terminals, min=1, max=32)
        num_terminals = torch.clip(num_terminals, min=1, max=256) # This is what it should be
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="center")

        terminal_timer.stop() if terminal_timer else None
        edge_timer.start() if edge_timer else None

        # generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets # (V, T, 2)
        terminal_distances = self.get_terminal_distances(terminal_positions, norm_order=self.distance_norm_order) # (V, T, V, T)
        edge_exists = get_distribution(**self.edge_dist).sample(terminal_distances) # (V, T, V, T)
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        # delete edges between same instance, among other things
        edge_exists = self.process_edge_matrix(edge_exists, is_source, num_terminals)
        self.connect_isolated_instances(edge_exists, terminal_distances)

        # convert to edge list and generate attributes
        edge_index, edge_attr = self.generate_edge_list(edge_exists, terminal_offsets)
        mask = placement.get_mask()
        if self.zero_edge_attr:
            edge_attr = 0 * edge_attr
        
        edge_timer.stop() if edge_timer else None

        data = Data(x=sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
        return positions, data
    
    def get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals, reference="center"):
        # NOTE here we assume reference point (for computing offset) is center of instance
        # NOTE outputs use whatever units are being used for sizes
        # x_sizes: (num_instances)
        # y_sizes: (num_instances)
        # max_num_terminals: int
        half_perim = (x_sizes + y_sizes)
                    
        terminal_locations = get_distribution("uniform", {"low": 0, "high": half_perim}).sample((max_num_terminals,)) # (max_term, num_instances)
        terminal_flip = get_distribution("bernoulli", {"probs": 0.5}).sample((max_num_terminals, x_sizes.shape[0])) # (max_term, num_instances)
        terminal_flip = (2 * terminal_flip) - 1

        x_sizes = x_sizes.unsqueeze(dim=0)
        y_sizes = y_sizes.unsqueeze(dim=0)
        boundary_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2) 
        boundary_offset_y = torch.clamp(terminal_locations-x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)

        boundary_offset_x = terminal_flip * boundary_offset_x
        boundary_offset_y = terminal_flip * boundary_offset_y

        boundary_offset = torch.stack((boundary_offset_x, boundary_offset_y), dim=-1).movedim(1, 0) # (num_instances, max_term, 2)
        
        # for some components, we want terminals on the interior, not the boundary
        if self.interior_terminals_dist is not None:
            sizes = torch.stack((x_sizes, y_sizes), dim=-1).squeeze(dim=0) # (num_instances, 2)
            gm_size = torch.sqrt(x_sizes * y_sizes).squeeze(dim=0)
            is_terminal_interior = get_distribution(**self.interior_terminals_dist).sample(gm_size).view(gm_size.shape[0], 1, 1)
            if self.interior_terminals_loc == "uniform":
                interior_offset = get_distribution("uniform", {"low": -sizes/2, "high": sizes/2}).sample((max_num_terminals,)) # (max_term, num_instances, 2)
                interior_offset = interior_offset.moveaxis(0, 1)
            elif self.interior_terminals_loc == "center":
                interior_offset = torch.zeros_like(boundary_offset)
            else:
                raise NotImplementedError
            terminal_offset = is_terminal_interior * interior_offset + (1-is_terminal_interior) * boundary_offset
        else:
            terminal_offset = boundary_offset
        
        if reference == "bottom_left":
            terminal_offset[:,:,0] += x_sizes/2
            terminal_offset[:,:,1] += y_sizes/2
        return terminal_offset

    def get_terminal_distances(self, terminal_positions, norm_order=1):
        # given global terminal positions (V, T, 2)
        # compute and return pairwise L1 distances between terminals
        # optionally, can specify x for Lx distance using norm_order
        if norm_order == "inf":
            norm_order = float(norm_order)
        V, T, _ = terminal_positions.shape
        t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
        t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
        delta_pos = t_pos_1 - t_pos_2 # (V, T, V, T, 2)
        distance = torch.norm(delta_pos, p=norm_order, dim=-1) # (V, T, V, T)
        return distance

    def process_edge_matrix(self, edge_exists, is_source, num_terminals):
        # edge_existence tensor (V, T, V, T)
        # is_source int tensor (V, T) (0=sink, 1=source)
        # num terminals int tensor (V)
        # process  as follows:
        # remove all edges ending in a source
        # remove all edges originating from a sink
        # remove all edges between same instance
        # remove all edges starting or ending in nonexistent terminal
        V, T, _, _ = edge_exists.shape
        assert is_source.shape == edge_exists.shape[:2]
        assert num_terminals.shape == (V,)

        # generate terminal filter
        terminal_filter = torch.zeros((V, T))
        for i, num_terminal in enumerate(num_terminals):
            terminal_filter[i, :num_terminal] = 1

        source_filter = (terminal_filter * is_source).view(V, T, 1, 1)
        sink_filter = (terminal_filter * (1-is_source)).view(1, 1, V, T)
        self_edge_filter = (1-torch.eye(V)).view(V, 1, V, 1)
        
        edges = edge_exists * source_filter
        edges = edges * sink_filter
        edges = edges * self_edge_filter
        return edges

    def connect_isolated_instances(self, edge_matrix, terminal_distances):
        # generate edges (IN-PLACE!) for disconnected instances
        # edge matrix has shape (V_src, T_src, V_dest, T_dest)
        # algorithm: for each instance with 0 degree, we connect edge from 0th terminal to closest available terminal
        # so that 0th terminal will become a sink
        # and we find the closest terminal that is:
        # - on a different vertex
        # - has out-degree
        V, T, _, _ = edge_matrix.shape
        out_degree = edge_matrix.sum(dim=(2,3)) # per terminal
        in_degree = edge_matrix.sum(dim=(0,1,3)) # per instance
        degree = out_degree.sum(dim=-1) + in_degree
        max_dist = 10+terminal_distances.max()
        for i in range(V):
            if degree[i] == 0: # isolated instance
                distances = terminal_distances[i, 0, :, :] # (V, T)
                distances = torch.where(out_degree > 0, distances, max_dist) # connect to existing nets only
                
                min_idx = torch.argmin(distances)
                instance_idx = min_idx // T
                terminal_idx = min_idx % T

                # connect to netlist
                edge_matrix[instance_idx, terminal_idx, i, 0] = 1

    def generate_edge_list(self, edge_exists, terminal_offsets):
        V, T, _, _ = edge_exists.shape
        edges = torch.nonzero(edge_exists) # (E, 4:v,t,v,t) int64
        edge_index_forward = edges[:,(0,2)]
        edge_index_reverse = edges[:,(2,0)]

        edge_attr_source = terminal_offsets[edges[:,0], edges[:,1], :] # (E, 2)
        edge_attr_sink = terminal_offsets[edges[:,2], edges[:,3], :] # (E, 2)
        edge_attr_forward = torch.concat((edge_attr_source, edge_attr_sink), dim=-1)
        edge_attr_reverse = torch.concat((edge_attr_sink, edge_attr_source), dim=-1)
        
        # create undirected edge index and attr
        edge_index = torch.concat((edge_index_forward, edge_index_reverse), dim=0).T # (2, E)
        edge_attr = torch.concat((edge_attr_forward, edge_attr_reverse), dim=0) # (E, 4)
        
        # return copy
        return edge_index.clone(), edge_attr.clone()

class V3(V2):
    def sample(self):
        # Generate stop density
        stop_density = get_distribution(**self.stop_density_dist).sample()

        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,))

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        # place samples individually
        placement = Placement()
        density = 0
        for (x_size, y_size) in zip(x_sizes, y_sizes):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([(x_size/2)-1.0, (y_size/2)-1.0]), "high": torch.tensor([1.0-(x_size/2), 1.0-(y_size/2)])}
            candidate_dist =  get_distribution("uniform", dist_params)
            
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()
                # print(instance_idx, attempt_num, candidate_pos)
                if placement.check_legality(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size)
                    break
            density += (x_size * y_size)/4.0
            if density >= stop_density:
                break
        
        positions = placement.get_positions()
        sizes = placement.get_sizes()
        num_instances = positions.shape[0]

        # sample number of terminals
        instance_area = sizes[:, 0] * sizes[:, 1]
        num_terminals = get_distribution(**self.num_terminals_dist).sample(instance_area).int()
        num_terminals = torch.clip(num_terminals, min=1, max=32)
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="center")

        # generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets # (V, T, 2)
        terminal_distances = self.get_terminal_distances(terminal_positions)
        # scale terminal distances by sizes of vertices
        scaled_terminal_distances = terminal_distances / self.get_terminal_dist_scale(instance_area)
        edge_exists = get_distribution(**self.edge_dist).sample(scaled_terminal_distances) # (V, T, V, T)
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        # delete edges between same instance, among other things
        edge_exists = self.process_edge_matrix(edge_exists, is_source, num_terminals)
        self.connect_isolated_instances(edge_exists, terminal_distances)

        # convert to edge list and generate attributes
        edge_index, edge_attr = self.generate_edge_list(edge_exists, terminal_offsets)
        mask = placement.get_mask()

        data = Data(x=sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
        
        return positions, data

    def get_terminal_dist_scale(self, instance_area):
        V, = instance_area.shape
        instance_scale = torch.sqrt(instance_area) # (V)
        source_scale = instance_scale.view(V, 1, 1, 1)
        dest_scale = instance_scale.view(1, 1, V, 1)
        p = 0.65
        dist_scale = torch.exp(p * torch.log(source_scale) + (1-p) * torch.log(dest_scale))
        return dist_scale

class V2_MaxSize(V2):
    def __init__(
            self, 
            cap_mode = "naive",
            size_upper_limit = 0.1,
            **kwargs,
        ):
        self.cap_mode = cap_mode
        self.size_upper_limit = size_upper_limit
        super().__init__(**kwargs)

    def sample(self):
        # Generate
        positions, data = super().sample()
        # clamp edge_attr and component sizes
        if self.cap_mode == "naive":
            data.x = torch.clamp(data.x, max = self.size_upper_limit)
            data.edge_attr = torch.clamp(data.edge_attr, max = self.size_upper_limit/2, min = -self.size_upper_limit/2)
        else:
            raise NotImplementedError
        return positions, data

class Flora():
    """
    Re-implementation of algorithm from Flora/GraphPlanner papers
    """
    def __init__(
            self,
            grid_size,
            density_dist,
            neighbors_dist,
            connectivity_dist,
            **kwargs
            ):
        self.grid_size = grid_size
        self.density_dist = density_dist
        self.neighbors_dist = neighbors_dist
        self.connectivity_dist = connectivity_dist
    
    def sample(
            self,
            size_dist_timer=None, # optional timers
            place_timer=None,
            terminal_timer=None,
            edge_timer=None,
            ):
        
        # We can precompute and cache a distance stencil maybe Too complicated, not worth the time

        # Generating object sizes and positions
        object_size = 2.0/self.grid_size        
        density = get_distribution(**self.density_dist).sample()
        grid_occupancy = torch.bernoulli(torch.full((self.grid_size, self.grid_size), density)).bool()

        x_coords = torch.linspace(-1+object_size/2, 1-object_size/2, self.grid_size)
        y_coords = torch.linspace(-1+object_size/2, 1-object_size/2, self.grid_size)
        coord_grid = torch.stack(torch.meshgrid(x_coords, y_coords, indexing="ij"), dim=-1) # (G, G, 2)

        obj_coords = coord_grid[grid_occupancy, :] # (V, 2)
        V, _ = obj_coords.shape
        obj_sizes = torch.full((V, 2), object_size)

        # Generating edge distributions TODO implement hierarchical gaussian
        num_neighbors = torch.round(torch.clip(get_distribution(**self.neighbors_dist).sample((V,)), min=1, max=V)).to(dtype=torch.int64)
        connectivity = torch.round(torch.clip(get_distribution(**self.connectivity_dist).sample((V, num_neighbors.max())), min=1, max=256)).to(dtype=torch.int64)
        
        # compute distances
        distances = torch.linalg.norm(obj_coords.view(V, 1, 2) - obj_coords.view(1, V, 2), dim=-1) # (V, V)
        
        _, distance_indices = torch.sort(distances, dim=1) # (V, V) each row is sorted

        # figuring out number of pairwise edges
        pairwise_edges = torch.zeros((V, V), dtype=torch.int64)
        for i in range(V):
            obj_neighbors = num_neighbors[i]
            obj_connectivity, _ = torch.sort(connectivity[i, :obj_neighbors]) # slice before sorting
            pairwise_edges[i, distance_indices[i, 1:obj_neighbors+1]] = obj_connectivity
        pairwise_edges = torch.floor((pairwise_edges + pairwise_edges.T)/2)

        # generating edge indices
        unique_edges, edge_counts = tgu.dense_to_sparse(torch.triu(pairwise_edges, diagonal=0))
        edge_indices = []
        for i, count in enumerate(edge_counts):
            count = count.int().item()
            edge_indices.append(unique_edges[:,i:i+1].repeat(1, count)) # (2, ...)
        edge_indices = torch.concatenate(edge_indices, dim=1) # (2, E_u)
        edge_indices = torch.concatenate((edge_indices, torch.flip(edge_indices, dims=(0,))), dim=1) # (2, E)
        _, E = edge_indices.shape
        edge_attr = torch.zeros((E, 4), dtype=torch.float32)

        mask = torch.zeros((V,)).bool()

        data = Data(x=obj_sizes, edge_index=edge_indices, edge_attr=edge_attr, is_ports=mask)
        return obj_coords, data

class Placement:
    def __init__(self, x = None, sizes = None, mask = None):
        # initializes empty placement, unless x, sizes, and mask are specified
        # x is predicted placements (V, 2)
        # attr is width height (V, 2)
        self.insts = []
        self.x = []
        self.y = []
        self.x_size = []
        self.y_size = []
        self.is_port = []
        if (x is not None) and (sizes is not None) and (mask is not None):
            for size, loc, is_ports in zip(sizes, x, mask):
                if not is_ports:
                    self.insts.append(
                        shapely.box(loc[0], loc[1], loc[0] + size[0], loc[1] + size[1])
                    )
            self.x = list(x[:,0])
            self.y = list(x[:,1])
            self.x_size = list(sizes[:,0])
            self.y_size = list(sizes[:,1])
            self.is_port = list(mask)

        self.chip = shapely.box(-1, -1, 1, 1)
        self.eps = 1e-8

    def check_legality(self, x_pos, y_pos, x_size, y_size, score=False):
        # checks legality of current placement (or optionally current placement with candidate)
        # x_pos, y_pos, x_size, y_size are floats
        # assumes given positions are center of instance
        # returns float with legality of placement (1 = bad, 0 = legal), or bool if score=False
        insts = self.insts + [shapely.box(x_pos - x_size/2, y_pos - y_size/2, x_pos + x_size/2, y_pos + y_size/2)]

        insts_area = sum([i.area for i in insts])
        insts_overlap = shapely.intersection(shapely.unary_union(insts), self.chip).area

        if score:
            return insts_overlap/insts_area
        else:
            return abs(insts_overlap - insts_area) < self.eps
    
    def commit_instance(self, x_pos, y_pos, x_size, y_size, is_port=False):
        # adds instance with specified params without checking if legal or not
        # assumes coordinates specified are center of instance
        if not is_port:
            self.insts.append(shapely.box(x_pos - x_size/2, y_pos - y_size/2, x_pos + x_size/2, y_pos + y_size/2))
        self.x.append(x_pos)
        self.y.append(y_pos)
        self.x_size.append(x_size)
        self.y_size.append(y_size)
        self.is_port.append(is_port)

    def get_density(self):
        insts_area = sum([i.area for i in self.insts])
        density = insts_area/self.chip.area
        return density
    
    def get_positions(self):
        # returns tensor(V, 2) of x,y placements
        positions = torch.stack((torch.tensor(self.x), torch.tensor(self.y)), dim=-1)
        return positions

    def get_sizes(self):
        # returns tensor(V, 2) of x,y sizes
        sizes = torch.stack((torch.tensor(self.x_size), torch.tensor(self.y_size)), dim=-1)
        return sizes
    
    def get_mask(self):
        mask = torch.tensor(self.is_port)
        return mask
    
class FastPlacement:
    # TODO
    def __init__(self):
        return

    def check_legality(self, pos, size):
        return
    
    def commit_instance(self, pos, size, is_port=False):
        return
    
    def get_positions(self):
        return
    
    def get_sizes(self):
        return
    
    def get_mask(self):
        return

def plot_placement(positions, sizes, name = "debug_placement"):
    picture = utils.visualize(positions, sizes)
    utils.debug_plot_img(picture, name)

def plot_sample(x, cond, name = "debug_placement", plot_edges=False):
    picture = utils.visualize_placement(x, cond, plot_edges=plot_edges)
    utils.debug_plot_img(picture, name)