from geoffrey_agent.player import Team as Geoffrey_Team
from image_agent.player import Team as Image_Team
from image_jurgen_agent.player import Team as Image_Jurgen_Team
from jurgen_agent.player import Team as Jurgen_Team
from yann_agent.player import Team as Yann_Team
from yoshua_agent.player import Team as Yoshua_Team

class DummyTeam(object):
    agent_type = 'image'
    dummy = True

    def __init__(self):
        self.num_players = None
        self.team = None
      
    def new_match(self, team, num_players):
        self.num_players = num_players
        self.team = team
    
    def act(self, player_states, player_images):
        return [dict()] * self.num_players


TEAM_DICT = {
    'we' : Image_Team,
    'geoffrey' : Geoffrey_Team,
    'jurgen' : Jurgen_Team,
    'image_jurgen' : Image_Jurgen_Team,
    'yann' : Yann_Team,
    'yoshua' : Yoshua_Team,
    'dummy' : DummyTeam,
}
