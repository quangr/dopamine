# coding=utf-8
# Copyright 2021 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for dopamine.jax.networks."""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax import networks
from flax import linen as nn
import jax
import numpy as onp
import jax.numpy as jnp

class NetworksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 32
    self._num_actions = 5
    self._input_shape = (84, 84, 4)
    self._batch_input_shape = (32, 84, 84, 4)
    self._rng = jax.random.PRNGKey(42)
    self._num_atoms = 21  # Rainbow/Quantile networks
    self._support = jax.numpy.linspace(-10, 10, self._num_atoms)  # Rainbow

  @parameterized.named_parameters(
      dict(testcase_name='DQN', network=networks.NatureDQNNetwork),
      dict(
          testcase_name='ClassicControlDQN',
          network=networks.ClassicControlDQNNetwork),
      dict(
          testcase_name='ClassicControlRainbow',
          network=networks.ClassicControlRainbowNetwork),
      dict(testcase_name='NDQN', network=networks.NonDecreasingQuantileNetwork),
      dict(testcase_name='Quantile', network=networks.QuantileNetwork),
      dict(testcase_name='Rainbow', network=networks.RainbowNetwork),
      dict(testcase_name='FullRainbow', network=networks.FullRainbowNetwork))
  def testOutputShape(self, network: nn.Module):
    kwargs = {}
    if network in [
        networks.FullRainbowNetwork, networks.RainbowNetwork,
        networks.ClassicControlRainbowNetwork, networks.QuantileNetwork
    ]:
      q_network = network(
          num_actions=self._num_actions, num_atoms=self._num_atoms)
      if network != networks.QuantileNetwork:
        kwargs = {'support': self._support}
    elif network in [networks.NonDecreasingQuantileNetwork]:
      q_network = network(
          num_actions=self._num_actions, num_quantiles=self._num_atoms)      
    else:
      q_network = network(num_actions=self._num_actions)

    x = onp.ones(self._input_shape)
    params = q_network.init(self._rng, x=x, **kwargs)
    def get_q_values(states):
      return q_network.apply(params, states, **kwargs).q_values
    if network in [networks.NonDecreasingQuantileNetwork]:
        q_network.apply(params,x, **kwargs)
        return
    get_q_values_batch = jax.vmap(get_q_values, in_axes=(0,))
    onp.testing.assert_equal(get_q_values(x).shape[0], self._num_actions)
    batch_q_values = get_q_values_batch(onp.ones(self._batch_input_shape))
    onp.testing.assert_equal(batch_q_values.shape,
                             (self._batch_size, self._num_actions))

class NDQNNetworksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 32
    self._num_actions = 5
    self._input_shape = (84, 84, 4)
    self._batch_input_shape = (32, 84, 84, 4)
    self._rng = jax.random.PRNGKey(42)
    self._num_atoms = 21  # Rainbow/Quantile networks
    self._support = jax.numpy.linspace(-10, 10, self._num_atoms)  # Rainbow

  def testclacQ(self):
    q_network = networks.NonDecreasingQuantileNetwork(
        num_actions=self._num_actions, num_quantiles=self._num_atoms,num_sample_quantiles=2)      
    x = onp.ones(self._input_shape)
    params = q_network.init(self._rng, x=x)
    def get_q_values(states):
      return q_network.apply(params, states)
    base,increase,Q=get_q_values(x)
    onp.testing.assert_equal(base.shape[0], self._num_actions)
    onp.testing.assert_equal(Q.shape[0], self._num_actions)
    onp.testing.assert_equal(increase.shape, (self._num_atoms,self._num_actions))
    get_q_values_batch = jax.vmap(get_q_values, in_axes=(0,))
    base,increase,_ = get_q_values_batch(onp.ones(self._batch_input_shape))
    onp.testing.assert_equal(base.shape,(self._batch_size, self._num_actions))
    onp.testing.assert_equal(increase.shape,(self._batch_size,self._num_atoms, self._num_actions))

  def testsampletau(self):
    q_network = networks.NonDecreasingQuantileNetwork(
        num_actions=self._num_actions, num_quantiles=2,num_sample_quantiles=2)      
    base=jnp.array([[1] for _ in range(5)]).T
    increase=jnp.array([[2,3] for _ in range(5)]).T
    onp.testing.assert_((q_network.interpolatetau(jnp.expand_dims(base,0),jnp.expand_dims(increase,0),jnp.array([1]),jnp.array([1/2]))==3).all())
    q_network.interpolatetau(jnp.expand_dims(base,0),jnp.expand_dims(increase,0),jnp.array([1]),jnp.array([1/3,2/3]))    


  def testsampletau_batch(self):
    x = onp.ones(self._input_shape)
    q_network = networks.NonDecreasingQuantileNetwork(
        num_actions=self._num_actions, num_quantiles=2,num_sample_quantiles=2)      
    params = q_network.init(self._rng, x=x)
    def get_q_values(states):
      return q_network.apply(params, states)
    get_q_values_batch = jax.vmap(get_q_values, in_axes=(0,))
    base,increase,_ = get_q_values_batch(onp.ones(self._batch_input_shape))    
    q_network.interpolatetau(base,increase,jnp.array([1]*self._batch_size),jnp.array([1/3,2/3]))    

  def testclacQ(self):
    q_network = networks.NonDecreasingQuantileNetwork(
        num_actions=self._num_actions, num_quantiles=2,num_sample_quantiles=2)      
    
    onp.testing.assert_(jnp.isclose(q_network.clacQ(jnp.array([[1] for _ in range(5)]).T,jnp.array([[2,3] for _ in range(5)]).T),13*0.499/2).all())
    
  def testclacQ_batch(self):
    x = onp.ones(self._input_shape)
    q_network = networks.NonDecreasingQuantileNetwork(
        num_actions=self._num_actions, num_quantiles=2,num_sample_quantiles=2)      
    params = q_network.init(self._rng, x=x)
    def get_q_values(states):
      return q_network.apply(params, states)
    get_q_values_batch = jax.vmap(get_q_values, in_axes=(0,))
    base,increase,_ = get_q_values_batch(onp.ones(self._batch_input_shape))    
    q_network.clacQ(base,increase)



if __name__ == '__main__':
  absltest.main()
