"""
Skill registry for managing and accessing skills in the movie recommendation system
"""
from typing import Dict, Type, Any
from .base_skill import BaseSkill


class SkillRegistry:
    """
    Registry for managing skills similar to Claude Skills registry.
    Allows registration and retrieval of skills by name.
    """
    
    def __init__(self):
        self._skills: Dict[str, Type[BaseSkill]] = {}
        self._instances: Dict[str, BaseSkill] = {}

    def register(self, name: str, skill_class: Type[BaseSkill]):
        """
        Register a skill class with a given name.
        
        Args:
            name: Name to register the skill under
            skill_class: The skill class to register
        """
        self._skills[name] = skill_class

    def get_skill(self, name: str, **kwargs) -> BaseSkill:
        """
        Get an instance of a registered skill.
        
        Args:
            name: Name of the skill to retrieve
            **kwargs: Arguments to pass to the skill constructor
            
        Returns:
            An instance of the requested skill
        """
        if name not in self._skills:
            raise ValueError(f"Skill '{name}' not registered")
            
        if name not in self._instances:
            self._instances[name] = self._skills[name](**kwargs)
        
        return self._instances[name]

    def list_skills(self) -> list:
        """
        List all registered skill names.
        
        Returns:
            List of registered skill names
        """
        return list(self._skills.keys())

    def execute_skill(self, name: str, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute a skill directly with the given state.
        
        Args:
            name: Name of the skill to execute
            state: State to pass to the skill
            **kwargs: Additional arguments for skill construction
            
        Returns:
            Updated state after skill execution
        """
        skill = self.get_skill(name, **kwargs)
        return skill.execute(state)


# Global registry instance
registry = SkillRegistry()