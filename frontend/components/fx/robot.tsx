"use client";
/* Pip, the robot companion. Adapted from a react-three-fiber robot hero: the scene is transparent,
 * fits whatever box it is placed in, follows the pointer anywhere on the page and has a
 * "thinking" mode that looks around while a calculation runs. No external assets are loaded. */
import * as React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";

export { ROBOT_NAME } from "./robot-name";

class HeartCurve extends THREE.Curve<THREE.Vector3> {
  constructor() {
    super();
  }
  getPoint(t: number, optionalTarget = new THREE.Vector3()) {
    t = t * Math.PI * 2;
    const x = 16 * Math.pow(Math.sin(t), 3);
    const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    return optionalTarget.set(x * 0.002, (y + 6) * 0.002, 0);
  }
}
const sharedHeartCurve = new HeartCurve();

/** Scales the robot so it fills the canvas whatever its aspect ratio. */
function FitGroup({ children, scale = 1 }: { children: React.ReactNode; scale?: number }) {
  const { viewport } = useThree();
  const s = Math.min(viewport.width / 1.6, viewport.height / 2.75) * scale;
  return <group scale={s} position={[0, -0.02 * s, 0]}>{children}</group>;
}

function GlassCapsule({ color, power, intensity }: { color: string; power: number; intensity: number }) {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const uniforms = useMemo(() => ({ color: { value: new THREE.Color("#ffffff") }, power: { value: 2.5 }, intensity: { value: 0.6 } }), []);
  useFrame(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.color.value.set(color);
      materialRef.current.uniforms.power.value = power;
      materialRef.current.uniforms.intensity.value = intensity;
    }
  });
  return (
    <mesh>
      <sphereGeometry args={[0.3, 64, 64, 0, Math.PI * 2, 0, Math.PI]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={`
          varying vec3 vNormal;
          varying vec3 vViewPosition;
          void main() {
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            vViewPosition = -mvPosition.xyz;
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * mvPosition;
          }
        `}
        fragmentShader={`
          uniform vec3 color;
          uniform float power;
          uniform float intensity;
          varying vec3 vNormal;
          varying vec3 vViewPosition;
          void main() {
            vec3 normal = normalize(vNormal);
            vec3 viewDir = normalize(vViewPosition);
            float fresnel = 1.0 - max(dot(viewDir, normal), 0.0);
            fresnel = pow(fresnel, power);
            gl_FragColor = vec4(color, fresnel * intensity);
          }
        `}
        transparent
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </mesh>
  );
}

const earBaseMat = new THREE.MeshStandardMaterial({ color: "#f0f0f0", roughness: 0.5 });
const earRingMat = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.3 });
const earCenterMat = new THREE.MeshStandardMaterial({ color: "#cccccc", roughness: 0.8 });
const antennaBaseMat = new THREE.MeshStandardMaterial({ color: "#999999", roughness: 0.4, metalness: 0.5 });
const antennaStickMat = new THREE.MeshStandardMaterial({ color: "#d0d0d0", roughness: 0.4, metalness: 0.2 });
const antennaTipMat = new THREE.MeshStandardMaterial({ color: "#ff3366", roughness: 0.2, toneMapped: false });

function RobotEar({ position, scale = 1, isLeft = false }: { position: [number, number, number]; scale?: number; isLeft?: boolean }) {
  const dir = isLeft ? -1 : 1;
  return (
    <group position={position} scale={scale}>
      <mesh rotation={[0, 0, Math.PI / 2]} material={earBaseMat}>
        <cylinderGeometry args={[0.04, 0.04, 0.025, 32]} />
      </mesh>
      <mesh position={[dir * 0.012, 0, 0]} rotation={[0, 0, Math.PI / 2]} material={earRingMat}>
        <torusGeometry args={[0.032, 0.008, 16, 32]} />
      </mesh>
      <mesh position={[dir * 0.012, 0, 0]} rotation={[0, 0, Math.PI / 2]} material={earCenterMat}>
        <cylinderGeometry args={[0.03, 0.03, 0.005, 32]} />
      </mesh>
      <group position={[dir * 0.015, 0.035, 0]} rotation={[-0.4, 0, 0]}>
        <mesh position={[0, 0.01, 0]} material={antennaBaseMat}>
          <cylinderGeometry args={[0.006, 0.008, 0.02, 16]} />
        </mesh>
        <mesh position={[0, 0.06, 0]} material={antennaStickMat}>
          <cylinderGeometry args={[0.003, 0.003, 0.1, 8]} />
        </mesh>
        <mesh position={[0, 0.11, 0]} material={antennaTipMat}>
          <sphereGeometry args={[0.006, 16, 16]} />
        </mesh>
      </group>
    </group>
  );
}

const eyeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(2, 2, 2), toneMapped: false, transparent: true });
const heartMat = new THREE.MeshBasicMaterial({ color: "#ff3366", toneMapped: false });

function RobotEye({ position, rotation, scale = 1, blinkDuration = 0.15, blinkCycle = 3.0, isLovedRef }: {
  position: [number, number, number]; rotation: [number, number, number]; scale?: number;
  blinkDuration?: number; blinkCycle?: number; isLovedRef: React.MutableRefObject<boolean>;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const normalEyesRef = useRef<THREE.Group>(null);
  const heartEyeRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!groupRef.current || !normalEyesRef.current || !heartEyeRef.current) return;
    const isHeart = isLovedRef.current;
    normalEyesRef.current.visible = !isHeart;
    heartEyeRef.current.visible = isHeart;
    const cycle = clock.getElapsedTime() % blinkCycle;
    let targetScaleY = 1;
    if (cycle < blinkDuration && !isHeart) {
      const blinkClose = Math.sin((cycle / blinkDuration) * Math.PI);
      targetScaleY = Math.max(0.05, 1.0 - blinkClose);
    }
    groupRef.current.scale.set(scale, scale * targetScaleY, scale);
  });

  const { topPath, bottomPath } = useMemo(() => {
    const w = 0.025, h = 0.035, r = 0.02, g = 0.005;
    const V = (x: number, y: number) => new THREE.Vector3(x, y, 0);
    const half = (sgn: 1 | -1) => {
      const p = new THREE.CurvePath<THREE.Vector3>();
      p.add(new THREE.LineCurve3(V(-w, sgn * g), V(-w, sgn * (h - r))));
      p.add(new THREE.QuadraticBezierCurve3(V(-w, sgn * (h - r)), V(-w, sgn * h), V(-w + r, sgn * h)));
      p.add(new THREE.LineCurve3(V(-w + r, sgn * h), V(w - r, sgn * h)));
      p.add(new THREE.QuadraticBezierCurve3(V(w - r, sgn * h), V(w, sgn * h), V(w, sgn * (h - r))));
      p.add(new THREE.LineCurve3(V(w, sgn * (h - r)), V(w, sgn * g)));
      return p;
    };
    return { topPath: half(1), bottomPath: half(-1) };
  }, []);

  return (
    <group ref={groupRef} position={position} rotation={rotation} scale={scale}>
      <mesh ref={heartEyeRef} visible={false} material={heartMat}>
        <tubeGeometry args={[sharedHeartCurve, 64, 0.0035, 8, true]} />
      </mesh>
      <group ref={normalEyesRef}>
        <mesh material={eyeMat}><tubeGeometry args={[topPath, 20, 0.0035, 8, false]} /></mesh>
        <mesh material={eyeMat}><tubeGeometry args={[bottomPath, 20, 0.0035, 8, false]} /></mesh>
      </group>
    </group>
  );
}

function generatePbrTexturesAsync(): Promise<{ colorMap: THREE.CanvasTexture; bumpMap: THREE.CanvasTexture }> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const size = 512;
      const canvasC = document.createElement("canvas");
      const canvasB = document.createElement("canvas");
      canvasC.width = canvasB.width = size;
      canvasC.height = canvasB.height = size;
      const ctxC = canvasC.getContext("2d");
      const ctxB = canvasB.getContext("2d");
      if (ctxC && ctxB) {
        ctxC.fillStyle = "#dcdcdc"; ctxC.fillRect(0, 0, size, size);
        ctxB.fillStyle = "#808080"; ctxB.fillRect(0, 0, size, size);
        for (let i = 0; i < 10000; i++) {
          const x = Math.random() * size, y = Math.random() * size, r = 0.5 + Math.random() * 1.5;
          const isDark = Math.random() > 0.15;
          ctxC.beginPath(); ctxC.arc(x, y, r, 0, Math.PI * 2); ctxC.fillStyle = isDark ? "#222222" : "#dddddd"; ctxC.fill();
          ctxB.beginPath(); ctxB.arc(x, y, r, 0, Math.PI * 2); ctxB.fillStyle = isDark ? "#000000" : "#ffffff"; ctxB.fill();
        }
      }
      const texC = new THREE.CanvasTexture(canvasC);
      const texB = new THREE.CanvasTexture(canvasB);
      texC.wrapS = texB.wrapS = THREE.RepeatWrapping;
      texC.wrapT = texB.wrapT = THREE.RepeatWrapping;
      texC.repeat.set(6, 3); texB.repeat.set(6, 3);
      texC.needsUpdate = texB.needsUpdate = true;
      resolve({ colorMap: texC, bumpMap: texB });
    }, 0);
  });
}

const NECK = { baseR: 0.215, baseH: -0.05, midR: 0.28, midH: 0.02, lipBottomR: 0.295, lipBottomH: 0.045, lipTopR: 0.27, lipTopH: 0.055, innerR: 0.1, innerDropH: 0.0 };
const BODY = { bodyBevelR: 0.235, bodyBevelY: 0.34, bodyBevelT: 0.025 };

function RobotPrototype({ color = "#c4c4c4", screenColor = "#00ffc6", screenGlow = 1.2, blinkCycle = 3.0, thinking = false }: {
  color?: string; screenColor?: string; screenGlow?: number; blinkCycle?: number; thinking?: boolean;
}) {
  const isLovedRef = useRef(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bodyRef = useRef<THREE.Group>(null);
  const headRef = useRef<THREE.Group>(null);
  const [textures, setTextures] = useState<{ colorMap: THREE.CanvasTexture; bumpMap: THREE.CanvasTexture } | null>(null);

  useFrame((state, delta) => {
    if (!bodyRef.current || !headRef.current) return;
    const dt = Math.min(delta, 0.1);
    const t = state.clock.getElapsedTime();
    // Thinking: Pip stays centred and facing forward; only the head glances gently from side to side.
    const tx = thinking ? 0 : state.pointer.x;
    const ty = thinking ? 0.1 : state.pointer.y;
    const glance = thinking ? Math.sin(t * 2.2) * 0.35 : 0;
    const bob = thinking ? Math.sin(t * 6) * 0.015 : Math.sin(t * 1.4) * 0.01;
    bodyRef.current.position.y = -0.3 + bob;

    const bodyRotY = -tx * 0.6, bodyRotX = -ty * 0.2, bodyRotZ = -tx * 0.1;
    const L = THREE.MathUtils.lerp;
    bodyRef.current.rotation.y = L(bodyRef.current.rotation.y, bodyRotY, 6 * dt);
    bodyRef.current.rotation.x = L(bodyRef.current.rotation.x, bodyRotX, 6 * dt);
    bodyRef.current.rotation.z = L(bodyRef.current.rotation.z, bodyRotZ, 6 * dt);
    headRef.current.rotation.y = L(headRef.current.rotation.y, tx * 1.1 + glance, 12 * dt);
    headRef.current.rotation.x = L(headRef.current.rotation.x, -ty * 0.3, 12 * dt);
  });

  useEffect(() => {
    let mounted = true;
    let maps: { colorMap: THREE.CanvasTexture; bumpMap: THREE.CanvasTexture } | null = null;
    generatePbrTexturesAsync().then((res) => {
      if (mounted) { maps = res; setTextures(res); } else { res.colorMap.dispose(); res.bumpMap.dispose(); }
    });
    return () => { mounted = false; maps?.colorMap.dispose(); maps?.bumpMap.dispose(); };
  }, []);

  const love = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    isLovedRef.current = true;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => { isLovedRef.current = false; }, 2000);
  };

  const neckProfile = useMemo(() => [
    new THREE.Vector2(NECK.innerR, NECK.baseH), new THREE.Vector2(NECK.baseR, NECK.baseH),
    new THREE.Vector2(NECK.midR, NECK.midH), new THREE.Vector2(NECK.lipBottomR, NECK.lipBottomH),
    new THREE.Vector2(NECK.lipTopR, NECK.lipTopH), new THREE.Vector2(NECK.innerR, NECK.lipTopH),
    new THREE.Vector2(NECK.innerR, NECK.lipTopH - NECK.innerDropH),
  ], []);
  const headMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#111111", roughness: 1.0, metalness: 0.0 }), []);

  if (!textures) return null;
  const shell = (
    <meshStandardMaterial color={color} map={textures.colorMap} bumpMap={textures.bumpMap} bumpScale={0.005} roughness={1.0} metalness={0} />
  );

  return (
    <group ref={bodyRef} position={[0, -0.3, 0]} onPointerDown={love}
      onPointerOver={() => (document.body.style.cursor = "pointer")} onPointerOut={() => (document.body.style.cursor = "auto")}>
      <mesh>
        <sphereGeometry args={[0.43, 64, 64, 0, Math.PI * 2, Math.PI * 0.15, Math.PI * 0.85]} />
        {shell}
      </mesh>
      <mesh position={[0, BODY.bodyBevelY, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[BODY.bodyBevelR, BODY.bodyBevelT, 32, 64]} />
        {shell}
      </mesh>
      <mesh position={[0, 0.38, 0]}>
        <latheGeometry args={[neckProfile, 64]} />
        {shell}
      </mesh>
      <group ref={headRef} position={[0, 0.6, 0]}>
        <mesh material={headMat}><sphereGeometry args={[0.28, 64, 64, 0, Math.PI * 2, 0, Math.PI]} /></mesh>
        <GlassCapsule color={screenColor} power={3.8} intensity={screenGlow} />
        <group position={[0, -0.02, 0.29]}>
          <RobotEye position={[-0.07, 0, 0]} rotation={[0, -0.2, 0]} scale={1.1} blinkDuration={0.45} blinkCycle={thinking ? 1.2 : blinkCycle} isLovedRef={isLovedRef} />
          <RobotEye position={[0.07, 0, 0]} rotation={[0, 0.2, 0]} scale={1.1} blinkDuration={0.45} blinkCycle={thinking ? 1.2 : blinkCycle} isLovedRef={isLovedRef} />
        </group>
        <RobotEar position={[-0.29, 0, 0]} isLeft scale={1.3} />
        <RobotEar position={[0.29, 0, 0]} scale={1.3} />
      </group>
    </group>
  );
}

/** Pip in a transparent canvas that fills its parent. */
export default function Robot({ thinking = false, className }: { thinking?: boolean; className?: string }) {
  const [source, setSource] = useState<HTMLElement | undefined>(undefined);
  useEffect(() => setSource(document.body), []);
  return (
    <div className={className} style={{ width: "100%", height: "100%" }}>
      {source && <Canvas
        camera={{ position: [0, 0.2, 6], fov: 40 }}
        gl={{ alpha: true, antialias: true }}
        dpr={[1, 2]}
        eventSource={source}
        eventPrefix="client"
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.85} />
        <hemisphereLight args={["#ffffff", "#30343a", 0.9]} />
        <directionalLight position={[2.5, 4, 5]} intensity={1.1} />
        <directionalLight position={[-4, 1, -3]} intensity={0.6} color="#7ff5dc" />
        <FitGroup>
          <RobotPrototype thinking={thinking} />
        </FitGroup>
      </Canvas>}
    </div>
  );
}
