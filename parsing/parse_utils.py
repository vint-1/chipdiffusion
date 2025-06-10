import re
from shapely import Polygon, unary_union, area
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from collections import OrderedDict
import pickle

class LibraryComponent:
    def __init__(self, name, is_macro, size):
        self.name = name
        self.is_macro = is_macro
        self.size = size

class DesignComponent:
    mult = 0 # shared class var WARNING THIS IS SHARED ONLY AMONG COMPONENTS IN A DESIGN TODO FIX THIS
    def __init__(self, name, typez, x=None, y=None, orientation=None):
        self.name = name
        self.type = typez
        if x and y:
            self.location =(float(x)/DesignComponent.mult, float(y)/DesignComponent.mult)
        self.orientation = orientation
        self.id = None
        self.pins = None
        self.is_macro = None
        self.pin_ids = None # maps pin names to unique IDs

    def add_size(self, inst_size):
        w, h = sT(inst_size, self.orientation)
        self.width = w
        self.height = h
        self.size =  (w, h)

    def add_pins(self, pins):
        '''
        Pin representation:
            {
                'pin_name1': (centroid_x, centroid_y)
            }
        Note: we use the centroid because pins sometimes have a shape that isn't a simple rectangle
        '''
        # transformation assumes the w and h have not been swapped
        if self.orientation in ('W', 'E', 'FW', 'FE'):
            size = (self.height,self.width)
        else:
            size = (self.width, self.height)
        
        
        pins = {p : T((round(pins[p].centroid.x, 5), round(pins[p].centroid.y, 5)), self.orientation, size) for p in pins}
        self.pin_ids = {pin_name: i for i, pin_name in enumerate(pins.keys())}
        self.pins = pins

class DefFile:
    """
    Used for input/outputs to def format.
    Currently only supports IBM benchmark format.
    """
    def __init__(self, path):
        with open(path, 'r') as f:
            input_text = f.read()
        
        components_start, components_end = self.get_section_indices(input_text, "COMPONENTS")
        nets_start, nets_end = self.get_section_indices(input_text, "NETS")
        design_start, design_end = self.get_section_indices(input_text, "DESIGN")
        
        self._text_sections = [
            input_text[:components_start],
            input_text[components_start:components_end],
            input_text[components_end:nets_start],
            input_text[nets_start:nets_end],
            input_text[nets_end:design_end],
            input_text[design_end:]
        ]
        self.COMPONENTS_IDX = 1
        self.NETS_IDX = 3

        self.components = self._text_sections[self.COMPONENTS_IDX].splitlines()
        self.netlines = self._text_sections[self.NETS_IDX].splitlines()

        self.canonicalize_components()
        self.placements = self.get_placements()

    def canonicalize_components(self):
        """
        Transform the component section such that each placement takes one line starting with '-' and ending with ';'
        """
        grouped_lines = []
        current_group = None
        for line in self.components[1:-1]: # remove first and last line for only placement info
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('-') and not current_group:
                # Start a new group
                current_group = line
            elif current_group:
                # Add to the current group
                current_group += " " + line
            
            if current_group.endswith(';'):
                # Finalize the current group if it ends with a semicolon
                grouped_lines.append(current_group)
                current_group = None

        # Handle any remaining group (optional, if the last group might not end with a ;)
        if current_group:
            grouped_lines.append(current_group)

        # put into form where p components are first and a components follow
        # Define a key function for sorting
        def sort_key(line):
            key = line.split()[1]  # Extract the second element (e.g., cluster{i}, p{i}, a{i})
            if key.startswith('cluster'):
                priority = 0  # Highest priority for 'cluster{i}'
            elif key.startswith('p'):
                priority = 1  # Second priority for 'p{i}'
            elif key.startswith('a'):
                priority = 2  # Lowest priority for 'a{i}'
            else:
                priority = 3  # Fallback for unknown cases, if any
            number = int(key[len(key.rstrip("0123456789")):])  # Extract the number part
            return (priority, number)  # Sort first by priority, then numerically
        
        grouped_lines = sorted(grouped_lines, key=sort_key)

        self.components[1:-1] = grouped_lines

    def get_placements(self):
        placements = []
        for line in self.components[1:-1]:
            parts = line.split()
            x, y = float(parts[6]), float(parts[7])
            placements.append((x, y))
        return np.array(placements)
    
    def set_placements(self, new_placements):
        for i, line in enumerate(self.components[1:-1]):
            parts = line.split()
            parts[6] = str(new_placements[i][0])
            parts[7] = str(new_placements[i][1])
            self.components[i+1] = " ".join(parts)
    
    def uncluster(self, cluster_map):
        placements = self.placements[cluster_map, :]
        self.set_placements(placements)

    def get_section_indices(self, text, section_name):
        """
        Get indices for start and end of section marked by 'SECTION_NAME ... END SECTION_NAME'
        """
        section_start = re.search(section_name, text).start()
        section_end = re.search(f"END\s+{section_name}", text).end()
        return section_start, section_end

    def update_components(self, clustered_id_map, clusters, nets):
        """
        Replaces COMPONENTS and NETS sections with new clustered components
        TODO handle case where placements are missing
        Inputs:
        - clustered_id_map: Dict obj_name -> clustered_component_idx
        - clusters: Dict cluster_idx -> Cluster object
        - nets: List[(str:src_obj, str:src_pin), List[(dest_obj, dest_pin), ...]]
        """
        # update components
        placed_component_pattern = re.compile(r"\s*-\s+(\S+)\s+(\S+)\s+(?:\+ SOURCE TIMING )?\+\s+(?:PLACED|FIXED|UNPLACED)\s+\(\s+(-?.?\d+\.?\d+) (-?.?\d+\.?\d+) \)\s+(\S+)(?:\s+;)?")
        new_components = ["COMPONENTS"]
        for i, cluster in clusters.items():
            # add clusters to beginning of new_components
            new_components.append(f"  -     cluster{i}       CLUSTER{i}  +    PLACED ( {cluster.position[0]*cluster.scale_factor} {cluster.position[1]*cluster.scale_factor} )  N  ;")
        for component_line in self.components:
            pcomponent_match = placed_component_pattern.match(component_line)
            if pcomponent_match:
                obj_name = pcomponent_match.group(1)
                if clustered_id_map[obj_name] not in clusters:
                    # retain only if object not clustered
                    new_components.append(component_line)
        # add start and end lines to components
        new_components[0]+=f" {len(new_components)-1} ;"
        new_components.append("END COMPONENTS")

        # update netlist
        new_nets = ["NETS"]
        for idx, (src_pin, dests) in enumerate(nets):
            new_nets.append(f"- net{idx} ")
            new_src_pin = self.map_pin(clustered_id_map, clusters, src_pin)
            new_nets.append(f"( {new_src_pin[0]} {new_src_pin[1]} )")
            for dest_pin in dests:
                new_dest_pin = self.map_pin(clustered_id_map, clusters, dest_pin)
                new_nets.append(f"( {new_dest_pin[0]} {new_dest_pin[1]} )")
            new_nets[-1]+="  ;"
        # add start and end lines to nets
        new_nets[0]+=f" {len(nets)} ;"
        new_nets.append("END NETS")

        self.components = new_components
        self.netlines = new_nets
        
    def map_pin(self, clustered_id_map, clusters, pin):
        if clustered_id_map[pin[0]] in clusters:
            cluster_idx = clustered_id_map[pin[0]]
            cluster = clusters[cluster_idx]
            cluster_name = f"cluster{cluster_idx}"
            return (cluster_name, cluster.get_pin_id(pin))
        else:
            return pin
        
    def fix_macros(self, fixed_macros):
        """
        Takes a list of macro names and the corresponding components to FIXED in the DEF
        """
        #TODO transform placement section to be consistent with newlines -- always between - and ;
        component_text = self.components[1:-1]
        for i, component_line in enumerate(component_text):
            parts = component_line.split()
            if parts[2] in fixed_macros:
                parts[4] = "FIXED"
            else:
                parts[4] = "PLACED"
            component_text[i] = " ".join(parts)
        self.components[1:-1] = component_text      

    def add_section(self, section_text):
        self._text_sections.insert(-1, section_text)
        
    def get_output(self):
        # updates self text_sections before generating outputs
        self._text_sections[self.COMPONENTS_IDX] = "\n".join(self.components)
        self._text_sections[self.NETS_IDX] = "\n".join(self.netlines)
        return "".join(self._text_sections)
    
    def write_output(self, path):
        with open(path, 'w') as f:
            f.write(self.get_output())

class LefFile:
    """
    Used for input/outputs to lef format.
    Currently only supports IBM benchmark format.
    TODO Ensure that cluster y size is bigger than site y size (ie. size of row)
    """
    def __init__(self, path):
        with open(path, 'r') as f:
            input_text = f.read()
        # split text into blocks based on MACRO keywords
        self.macro_text_blocks, self.macro_text_block_indices = self.split_text_sections(input_text, "MACRO")
    
    def split_text_sections(self, text, pattern):
        """
        Looks through text (given as string) and finds all the blocks that satisfy <pattern> <id> ... END <id>
        - text: str
        - pattern: str eg. "MACRO"
        returns:
        - text_blocks: List[str]
        - pattern_indices: OrderedDict{str(id): int(index in text_blocks that satisfy pattern)}
        """
        pattern_regex = f"{pattern}\s+(\S+)\s*\n"
        
        text_blocks = []
        pattern_indices = OrderedDict()
        while True:
            match = re.search(pattern_regex, text)
            if match is None:
                break

            block_id = match.group(1)
            block_start = match.start()
            if block_start>0:
                text_blocks.append(text[:block_start])
                text = text[block_start:]
            
            pattern_end_regex = f"END\s+{block_id}"
            block_end = re.search(pattern_end_regex, text).end()
            pattern_indices[block_id] = len(text_blocks)

            text_blocks.append(text[:block_end])
            text = text[block_end:]
        text_blocks.append(text)

        return text_blocks, pattern_indices

    def update_components(self, component_map, clustered_id_map, clusters, nets):
        """
        Replaces the appropriate MACRO definitions with new clustered components
        Inputs:
        - component_map: Dict def_obj_name -> lef_object_name (original objects)
        - clustered_id_map: Dict obj_name -> clustered_component_idx
        - clusters: Dict cluster_idx -> Cluster object
        - nets: List[(str:src_obj, str:src_pin), List[(dest_obj, dest_pin), ...]]
        """
        # iterate through clusters
        # and generate macro block text
        cluster_text_blocks = [self.cluster_to_text(f"CLUSTER{i}", cluster, component_map) for i, cluster in clusters.items()]

        # construct new list of macro text blocks
        remaining_blocks = set([v for k, v in component_map.items() if (clustered_id_map[k] not in clusters)])
        delete_blocks = set(self.macro_text_block_indices.keys()) - remaining_blocks

        # add the cluster blocks at the start
        new_text_blocks = [self.macro_text_blocks[0]]
        new_text_block_indices = OrderedDict()
        for cluster_text_block in cluster_text_blocks:
            new_text_block_indices[re.search("^\s*MACRO\s+(CLUSTER\S+)",cluster_text_block).group(1)] = len(new_text_blocks)
            new_text_blocks.append(cluster_text_block)
            new_text_blocks.append("\n\n")
        
        # delete the SC blocks and copy original macro text blocks
        for block_name, block_idx  in self.macro_text_block_indices.items():
            if block_name in remaining_blocks:
                new_text_block_indices[block_name] = len(new_text_blocks)
                new_text_blocks.append(self.macro_text_blocks[block_idx])
                new_text_blocks.append("\n\n")
        new_text_blocks.append(self.macro_text_blocks[-1])

        self.macro_text_blocks = new_text_blocks
        self.macro_text_block_indices = new_text_block_indices

    def cluster_to_text(self, cluster_name, cluster, component_map):
        """
        Convert cluster object into single block of DEF-formatted text
        returns string
        """

        # generate cluster header
        base_component_id = component_map[next(iter(cluster.pins.keys()))]
        base_text = self.macro_text_blocks[self.macro_text_block_indices[base_component_id]]
        base_text_blocks, _ = self.split_text_sections(base_text, "PIN")
        base_header_lines = base_text_blocks[0].splitlines()[1:]
        cluster_lines = [ 
            f"MACRO {cluster_name}",
            f"\tCLASS BLOCK ;",
            f"\tSIZE {cluster.shape[0]} BY {cluster.shape[1]} ;",
        ]
        for base_header_line in base_header_lines:
            if not re.search("MACRO|CLASS|SIZE|SYMMETRY", base_header_line) and not base_header_line.isspace():
                cluster_lines.append(base_header_line)
        
        # prepare common info for all pins
        pin_location = (
            cluster.shape[1] * 0.49, # y_min
            cluster.shape[0] * 0.49, # x_min
            cluster.shape[1] * 0.51, # y_max
            cluster.shape[0] * 0.51, # x_max
        )

        # add data for pins
        for def_component_name, component_pin_names in cluster.pins.items():
            component_id = component_map[def_component_name]
            component_text = self.macro_text_blocks[self.macro_text_block_indices[component_id]]
            component_text_blocks, component_text_block_indices = self.split_text_sections(component_text, "PIN")
            # iterate over all external-connecting pins for a given component in cluster
            for component_pin_name in component_pin_names:
                cluster_pin_id = cluster.pin_ids[(def_component_name, component_pin_name)]
                original_pin_lines = component_text_blocks[component_text_block_indices[component_pin_name]].splitlines()
                pin_lines = [f"\tPIN {cluster_pin_id}"]
                # Copy lines from original pin to new; place pin positions in the center
                for original_pin_line in original_pin_lines[1:]:
                    if not re.search(f"(.\s*RECT\s+)|(.\s*END\s+{component_pin_name}\s*$)", original_pin_line) and not original_pin_line.isspace():
                        pin_lines.append(original_pin_line)
                    elif match := re.search("(\s+)RECT", original_pin_line):
                        pin_lines.append(
                            f"{match.group(1)}RECT {pin_location[0]} {pin_location[1]} {pin_location[2]} {pin_location[3]} ;"
                        )
                pin_lines.append(f"\tEND {cluster_pin_id}")
                cluster_lines.extend(pin_lines)
        cluster_lines.append(f"END {cluster_name}")
        cluster_text = "\n".join(cluster_lines)
        return cluster_text

    def get_output(self):
        return "".join(self.macro_text_blocks)

    def write_output(self, path):
        with open(path, 'w') as f:
            f.write(self.get_output())

# To rotate the location on the chip
def lT(loc, orientation, chip_size, comp_size):
    loc = T(loc, orientation, chip_size)
    if orientation == 'S':
        return (loc[0] - comp_size[0], loc[1] - comp_size[1])
    elif orientation == 'W':
        return (loc[0] - comp_size[1], loc[1])
    elif orientation == 'E':
        return (loc[0], loc[1] - comp_size[0])
    elif orientation == 'FN':
        return (loc[0] - comp_size[0], loc[1])
    elif orientation == 'FS':
        return (loc[0], loc[1] - comp_size[1])
    elif orientation == 'FW':
        return (loc[0], loc[1])
    elif orientation == 'FE':
        return (loc[0] - comp_size[1], loc[1] - comp_size[0])
    else:
        return loc

# rotate pin coordinates
def T(loc, orientation, size):
    width, height = size
    if orientation == 'N':
        transformation = lambda x, y : (x, y)
    elif orientation == 'S':
        transformation = lambda x, y : (width - x, height - y)
    elif orientation == 'W':
        transformation = lambda x, y: (height - y, x)
    elif orientation == 'E':
        transformation = lambda x, y: (y, width - x)
    elif orientation == 'FN':
        transformation = lambda x, y: (width - x, y)
    elif orientation == 'FS':
        transformation = lambda x, y: (x, height - y)
    elif orientation == 'FW':
        transformation = lambda x, y: (y, x)
    elif orientation == 'FE':
        transformation = lambda x, y: (height - y, width - x)
    else:
        transformation = lambda x, y : (x, y)
    
    # The IBM benchmarks means that we cannot assume this
    # if any([pos < 0 for pos in transformation(*loc)]):
    #     import ipdb; ipdb.set_trace()
    
    return transformation(*loc)

# rotate size
def sT(loc, orientation):
    if orientation in ('W', 'E', 'FW', 'FE'):
        return (lambda x, y: [y, x])(*loc)
    else:
        return (lambda x, y: [x, y])(*loc)

def parse_lef_file(lef_filename):
    """
    Reads LEF file and creates pythonic structures
    lef_filename: str or path
    returns:
    - Components: dict(name: LibraryComponent)
    - Pins: dict(...)
    """
    inst_comps = {}
    inst_pins = {}
    with open(lef_filename, 'r') as lef_file:
        lef_file_data = lef_file.read()

    # Parse important info in the header
    mult = re.search(r"MICRONS\s+([0-9]+) ;", lef_file_data.split('MACRO')[0]).group(1)

    # Parse each instance
    inst_type_pattern = re.compile(r"\s+CLASS BLOCK")
    inst_size_pattern = r"SIZE ([0-9.]+) BY ([0-9.]+)"
    unwanted_pins = ['VPWR', 'VPB', 'VNB', 'VGND', 'VSSD', 'VSSA', 'VDD', 'VSS']
    # import ipdb;ipdb.set_trace()
    # import pdb; pdb.set_trace()
    for macro in lef_file_data.split('MACRO')[1:]:
        name = ""
        macro_header = macro.split('PIN')[0]
        
        
        if macro_header:
            lines = macro_header.splitlines()
            name = lines[0].strip()
            is_macro = bool(inst_type_pattern.match(lines[1])) or bool(inst_type_pattern.match(lines[4]))
            sizes = re.search(inst_size_pattern, macro_header)
            size = (float(sizes.group(1)), float(sizes.group(2)))
            # if name == 'fakeram45_256x64':
            #     import ipdb; ipdb.set_trace()
            inst_comps[name] = LibraryComponent(name, is_macro, size)

        inst_pins[name] = {}
        for line in macro.split('PIN')[1:]:
            # sometimes OBS is listed at the end of the pin list
            # not sure what it is, but needs to be omitted for correctness
            line = line.split('OBS')[0]
            # always the next word after 'PIN'
            pin_name = line.splitlines()[0].strip()
            if pin_name in unwanted_pins:
                continue
            # Most extreme matching case: 6.93889e-19
            rects = re.findall(r"RECT\s+([0-9.e\-]+) ([0-9.e\-]+) ([0-9.e\-]+) ([0-9.e\-]+)", line, re.DOTALL)
            if rects == []:
                continue
            rects = [[float(c) for c in r] for r in rects]
            rects = [Polygon([(r[0], r[1]), (r[0], r[3]), (r[2], r[3]), (r[2], r[1])]) for r in rects]
            # inst_pins[name][pin_name] = max(rects, key=area)
            inst_pins[name][pin_name] = unary_union(rects)

            if (inst_pins[name][pin_name].is_empty):
                print(f'[WARN]: Macro {name} at {pin_name} has no shape definition')
                inst_pins[name].pop(pin_name, None)

    return inst_comps, inst_pins

def parse_def_file(location_filename, only_netlist=True, netlist_filename=""):
    # parse location file for final inst locations, but not netlist bc it has buffers
    # parse netlist file for netlist, but not inst locations bc they are not finalized
    with open(location_filename, 'r') as def_file:
        location_data = def_file.read()
    if netlist_filename:
        with open(netlist_filename, 'r') as def_file:
            netlist_data = def_file.read()
    else:
        netlist_data = location_data
    location_data_split = re.split(r'COMPONENTS|PINS|NETS', location_data)
    netlist_data_split = re.split(r'COMPONENTS|PINS|NETS', netlist_data)
    assert(len(location_data_split) == len(netlist_data_split))

    # parse header
    multiplier_pattern = re.search(r"MICRONS ([0-9]+)", location_data_split[0])
    DesignComponent.mult = float(multiplier_pattern.group(1))
    chip_size_pattern = re.search(r"DIEAREA \( (-?[0-9.]+) (-?[0-9.]+) \) \( (-?[0-9.]+) (-?[0-9.]+) \)", location_data_split[0])
    chip_size = [float(chip_size_pattern.group(i))/DesignComponent.mult for i in range(1, 5)]
    
    if len(location_data_split) == 7: # TODO get rid of these magic numbers
        insts_loc_section = location_data_split[1]
        insts_net_section = netlist_data_split[1]
        ports_section = location_data_split[3]
        nets_section = netlist_data_split[5]

        components = {}
        parse_insts(insts_loc_section, components, only_netlist, insts_net_section)
        parse_ports(ports_section, components, only_netlist)

        nets_graph, nets_list, nets_routing = parse_edges(nets_section, do_routing=False)
    elif len(location_data_split) == 5: # For current IBM benchmarks
        insts_loc_section = location_data_split[1]
        insts_net_section = netlist_data_split[1]
        nets_section = netlist_data_split[3]

        components = {}
        parse_insts(insts_loc_section, components, only_netlist, insts_net_section)

        nets_graph, nets_list, nets_routing = parse_edges(nets_section, do_routing=False)

    return components, nets_graph, nets_list, nets_routing, chip_size

def parse_layers(def_data):
    layers = re.findall('LAYER (\S+) ;', def_data)
    return set(layers)

def parse_insts(location_data, components, only_netlist, netlist_data=None):
    # Define patterns for layers and components
    # placed_component_pattern = re.compile(r"-\s+(\S+)\s+(\S+)\s+(?:\+ SOURCE TIMING \+|\+) PLACED \( (\d+) (\d+) \)\s+(\S+)")
    # placed_component_pattern = re.compile(r"\s*-\s+(\S+)\s+(\S+)\s+(?:\+ SOURCE TIMING )?\+\s+(?:PLACED|FIXED|UNPLACED) \( (-?.?\d+) (-?.?\d+) \)\s+(\S+)(?:\s+;)?")
    placed_component_pattern = re.compile(r"\s*-\s+(\S+)\s+(\S+)\s+(?:\+ SOURCE TIMING )?\+\s+(?:PLACED|FIXED|UNPLACED)\s+\(\s+(-?\.?\d+\.?\d*) (-?\.?\d+\.?\d*) \)\s+(\S+)(?:\s+;)?")
    component_names = []

    if only_netlist:
        '''
        Without the locations, components in the def look like the following:
        - inst_name inst_type
        '''
        for line in location_data.splitlines():
            if (len(line.split()) == 3):
                _, name, type = line.split()
                components[name] = DesignComponent(name, type)
        
        print(len(components))
        return components

    if netlist_data is not None:
        for line in netlist_data.split(";"):
            pcomponent_match = placed_component_pattern.match(line)
            if pcomponent_match:
                component_names.append(pcomponent_match.group(1))
    
    for line in location_data.split(";"):
        pcomponent_match = placed_component_pattern.match(line)
        if pcomponent_match and (netlist_data is None or pcomponent_match.group(1) in component_names):
            traits = [pcomponent_match.group(i) for i in range(1,6)]
            components[traits[0]] = (DesignComponent(*traits))

def parse_ports(def_data, components, only_netlist):
    ports = def_data.split('- ')[1:]
    power_pins = ['VDD', 'VSS']
    
    for port in ports:
        # last port is off by one
        last_port = port == ports[-1]
        port = port.split()
        if port[0] in power_pins:
            continue
        # name, typez, x, y, orientation
        if only_netlist:
            traits = (port[0], 'PORT')
        else:
            traits = (port[0], 'PORT', port[-5 - last_port], port[-4 - last_port], port[-2 - last_port])
        components[traits[0]] = DesignComponent(*traits)

def parse_edges(def_data, do_routing=True):
    edges = []
    routing_nets = {}
    nets = [] # each net is tuple(source: tuple(obj, pin), destinations: list[tuple(obj, pin), ...])

    routing_pattern = re.compile('\s+\S+ (\S+) \( ([0-9*]+) ([0-9*]+)(?: 0)? \) \( ([0-9*]+) ([0-9*]+)(?: 0)? \)*')

    for net in def_data.split('- ')[1:]:
        if not net or 'UNCONNECTED' in net:
            continue
        net_name = net.split()[0]
        # skip component if completely unconnected
        is_routed = net.split('ROUTED')
        endpoints = is_routed[0]
        insts = re.findall('\( (\S+) (\S+) \)', endpoints, re.DOTALL)
        if insts:
            # Use pin name as inst name
            driver = insts[0][0] if insts[0][0] != "PIN" else insts[0][1]
            e = [(driver, i[0] if i[0] != "PIN" else i[1], {'u_pin': insts[0][1], 'v_pin': i[1]}) for i in insts[1:]]
            edges += e
            
            # TODO handle edge case where "PIN" appears
            assert all([not "PIN" in obj for obj,_ in insts]), "Unhandled case when parsing def: PIN keyword in netlist"
            nets.append((insts[0], list(insts[1:])))

        if do_routing and len(is_routed) == 2:
            routing = is_routed[1]
            net_list = [(routing_pattern.match(line).group(i) for i in range(1, 6)) for line in routing if routing_pattern.match(line)]
            routing_nets[net_name] = net_list

    return edges, nets, routing_nets

def update_design_components(components, lib_components, pins):
    """
    Updates DesignComponents with LEF data
    Note: performs updates in-place
    """
    id = 0
    for i in components:
        if components[i].type == 'PORT':
            components[i].add_size((1, 1))
            components[i].add_pins({components[i].name: Polygon([(0,0), (0, 1), (1, 1), (1, 0)])})
            components[i].is_macro = False
        else:
            # print(components[i].type)
            components[i].add_size(lib_components[components[i].type].size)
            components[i].add_pins(pins[components[i].type])
            components[i].is_macro = lib_components[components[i].type].is_macro
        components[i].id = id
        id += 1
    return components

def to_graph(components, nets_graph):
    graph = nx.Graph()
    # convert all ids to integers
    # edges attr instance 1 id, instance 2 id, pin1 x, pin1 y, pin2 x, pin2 y
    edges = [(components[u].id, components[v].id, {
                                        'u_pinx': components[u].pins[attr['u_pin']][0],
                                        'u_piny': components[u].pins[attr['u_pin']][1],
                                        'v_pinx': components[v].pins[attr['v_pin']][0],
                                        'v_piny': components[v].pins[attr['v_pin']][1]}) for u, v, attr in nets_graph]
    # edges += [(components[v].id, components[u].id, {
    #                                     'u_pinx': components[v].pins[attr['v_pin']][0],
    #                                     'u_piny': components[v].pins[attr['v_pin']][1],
    #                                     'v_pinx': components[u].pins[attr['u_pin']][0],
    #                                     'v_piny': components[u].pins[attr['u_pin']][1]}) for u, v, attr in nets_graph]
    graph.add_edges_from(edges)

    components_list = list(components.keys())
    for id in graph.nodes:
        # dereference id
        name = components_list[id]
        graph.nodes[id]['width'] = components[name].width
        graph.nodes[id]['height'] = components[name].height
    return graph

def to_torch_data(components, nets_graph, r, chip_size):
    components_list = list(components.keys())
    data_x = torch.tensor([sT(components[c].size, r) for c in components_list])
    data_ports = torch.tensor([components[c].type == 'PORT' for c in components_list])
    data_macros = torch.tensor([components[c].is_macro for c in components_list])

    data_edge_index_uv = torch.tensor([[components[u].id, components[v].id] for u, v, _ in nets_graph]).T
    data_edge_index_vu = torch.tensor([[components[v].id, components[u].id] for u, v, _ in nets_graph]).T
    data_edge_index = torch.hstack((data_edge_index_uv, data_edge_index_vu)) # (2, 2E)

    data_pin_id_uv = torch.tensor([[components[u].pin_ids[attr['u_pin']], components[v].pin_ids[attr['v_pin']]] for u, v, attr in nets_graph])
    data_pin_id_vu = torch.tensor([[components[v].pin_ids[attr['v_pin']], components[u].pin_ids[attr['u_pin']]] for u, v, attr in nets_graph])
    data_edge_pin_id = torch.vstack((data_pin_id_uv, data_pin_id_vu)) # (2E, 2)

    data_edge_attr_uv = torch.tensor([[*T(components[u].pins[attr['u_pin']], r, components[u].size),
                                       *T(components[v].pins[attr['v_pin']], r, components[v].size)] for u, v, attr in nets_graph])
    data_edge_attr_vu = torch.tensor([[*T(components[v].pins[attr['v_pin']], r, components[v].size),
                                       *T(components[u].pins[attr['u_pin']], r, components[u].size)] for u, v, attr in nets_graph])
    data_edge_attr = torch.vstack((data_edge_attr_uv, data_edge_attr_vu)) # (2E, 4)

    return Data(
        x=data_x, 
        is_ports=data_ports, 
        is_macros=data_macros, 
        edge_index=data_edge_index, 
        edge_attr=data_edge_attr, 
        edge_pin_id=data_edge_pin_id,
        chip_size=chip_size
    )
    
def get_locations(components, chip_size, r='N'):
    # TODO check for bugs
    mat = np.zeros((len(components), 2))

    for row, comp in zip(mat, components.keys()):
        # if (components[comp].location[0] < chip_size[0] or components[comp].location[1] < chip_size[1] or 
            # components[comp].location[0] > chip_size[2] or components[comp].location[1] > chip_size[3]):
            # import ipdb; ipdb.set_trace()

        row[0], row[1] = lT(components[comp].location, r, (chip_size[2], chip_size[3]), components[comp].size)
    return mat

def copy_ref_placement(placement_components, target_components, placement_mult=1.0, target_mult=1.0):
    """
    Copies (x, y) placement from placement_components to target_components
    """
    for k, placement_component in placement_components.items():
        scale_factor = placement_mult / target_mult
        new_placement = (scale_factor * placement_component.location[0], scale_factor * placement_component.location[1])
        target_components[k].location = new_placement

def open_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)