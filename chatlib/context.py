from .common import *

class Context:
    """
    Keeps track of IDs and can convert between clean and messy IDs.
    """
    def __init__(self):        
        self.id_map = {}
        self.id_map_r = {}
        self.id_map_counter = 1

    def clean_id(self, id):
        if isinstance(id, ObjectId):
            id = str(id)

        if id not in self.id_map:
            self.id_map[id] = self.id_map_counter
            self.id_map_r[self.id_map_counter] = id
            self.id_map_counter += 1
                    
        return self.id_map[id]

    def messy_id(self, clean_id):
        clean_id = int(clean_id)
        return ObjectId(self.id_map_r[clean_id])
    

    def cleanify_markdown(self, text):
        """
        Expanding links like [text](messy_id) with [text](clean_id).
        """
        def replace_id(match):
            #logger.debug(f'processing markdown match: {match.group(1)}, {match.group(2)}')
            messy_id = match.group(2)
            clean_id = self.clean_id(messy_id)
            return f"[{match.group(1)}]({clean_id})"

        text = re.sub(r"\[([^\[\]]*)\]\(([^\(\)]+)\)", replace_id, text)
        return text
    

    def messify_markdown(self, text):
        """
        Expanding links like [text](clean_id) with [text](messy_id).
        Also, returning the messy_ids found.
        """
        
        messy_ids = []
        def replace_id(match):
            #logger.debug(f'processing markdown match: {match.group(1)}, {match.group(2)}')
            clean_id = match.group(2)
            messy_id = self.messy_id(clean_id)
            messy_ids.append(messy_id)
            return f"[{match.group(1)}]({messy_id})"

        # [arbitrary text](clean_id) is the input
        text = re.sub(r"\[([^\[\]]*)\]\(([^\(\)]+)\)", replace_id, text)
        return text, messy_ids
