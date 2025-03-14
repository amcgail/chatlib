"""Tests for the Actor registry system."""

import pytest
from bson import ObjectId
from chatlib.convo import Actor, register_actor, _actor_registry

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the actor registry before each test."""
    saved_base_actor = _actor_registry.get('Actor')
    _actor_registry.clear()
    if saved_base_actor:
        _actor_registry['Actor'] = saved_base_actor
    yield

def test_base_actor_registration():
    """Test that the base Actor class is automatically registered."""
    assert 'Actor' in _actor_registry
    assert _actor_registry['Actor'] == Actor

def test_custom_actor_registration():
    """Test registering a custom actor type."""
    @register_actor
    class CustomActor(Actor):
        def custom_method(self):
            return "custom method called"
    
    assert 'CustomActor' in _actor_registry
    assert _actor_registry['CustomActor'] == CustomActor

def test_actor_persistence():
    """Test saving and loading actors with proper type restoration."""
    @register_actor
    class CustomerServiceActor(Actor):
        def handle_complaint(self, complaint):
            self['last_complaint'] = complaint
            return f"Handling complaint: {complaint}"
    
    # Create and save actor
    actor = CustomerServiceActor(can_leave=True, name="Support Agent")
    actor_id = actor.save()
    
    # Load actor and verify type and data
    loaded_actor = Actor.load(actor_id)
    assert isinstance(loaded_actor, CustomerServiceActor)
    assert loaded_actor['name'] == "Support Agent"
    assert loaded_actor.can_leave == True
    
    # Test custom method
    response = loaded_actor.handle_complaint("slow service")
    assert "Handling complaint: slow service" in response
    assert loaded_actor['last_complaint'] == "slow service"

def test_unregistered_actor_load():
    """Test that loading an unregistered actor type raises an error."""
    # Create an actor of a type that will not be registered
    @register_actor
    class TemporaryActor(Actor):
        pass
    
    actor = TemporaryActor(name="Temp")
    actor_id = actor.save()
    
    # Clear registry and try to load
    _actor_registry.clear()
    _actor_registry['Actor'] = Actor  # Restore base Actor
    
    with pytest.raises(ValueError) as exc_info:
        Actor.load(actor_id)
    
    assert "not found in registry" in str(exc_info.value)
    assert "Did you forget to register it" in str(exc_info.value)

def test_actor_attribute_persistence():
    """Test that actor attributes are properly persisted and loaded."""
    @register_actor
    class ConfigurableActor(Actor):
        def update_config(self, key, value):
            self[key] = value
    
    # Create and configure actor
    actor = ConfigurableActor()
    actor.update_config('api_key', '12345')
    actor.update_config('model', 'gpt-4')
    actor_id = actor.save()
    
    # Load and verify
    loaded_actor = Actor.load(actor_id)
    assert loaded_actor['api_key'] == '12345'
    assert loaded_actor['model'] == 'gpt-4'
    
    # Update and verify persistence
    loaded_actor.update_config('model', 'gpt-5')
    
    # Load again to verify persistence
    reloaded_actor = Actor.load(actor_id)
    assert reloaded_actor['model'] == 'gpt-5' 