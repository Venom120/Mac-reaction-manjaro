class ReactionManager:
    def __init__(self):
        self._is_reaction_active = False

    def is_reaction_active(self):
        """Checks if any reaction effect is currently active."""
        return self._is_reaction_active

    def set_reaction_active(self, active):
        """Sets the state of whether a reaction is active."""
        self._is_reaction_active = active