// SPDX-License-Identifier: GPL-2.0
/*
 * Out-of-tree RK3588 RKNPU driver for mainline Linux 7.x (atk-dlrk3588-bsp).
 *
 * Implements the /dev/rknpu miscdevice ioctl ABI required by Rockchip's
 * librknnrt.so (v1.5.2+) against the mainline rk3588-base.dtsi
 * "rockchip,rk3588-rknn-core" device-tree nodes (npu@fdab0000, npu@fdac0000,
 * npu@fdad0000), including live DT node activation when booted with a DTB
 * where the NPU nodes were left status = "disabled".
 */

#include <linux/anon_inodes.h>
#include <linux/atomic.h>
#include <linux/clk.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/gfp.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/ioctl.h>
#include <linux/kernel.h>
#include <linux/kref.h>
#include <linux/ktime.h>
#include <linux/list.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_platform.h>
#include <linux/platform_device.h>
#include <linux/pm_runtime.h>
#include <linux/reset.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <drm/drm.h>

#define DRIVER_NAME "rknpu"
#define DRIVER_MAJOR 0
#define DRIVER_MINOR 9
#define DRIVER_PATCHLEVEL 6
#define RKNPU_DRV_VERSION_CODE \
	((DRIVER_MAJOR)*10000 + (DRIVER_MINOR)*100 + (DRIVER_PATCHLEVEL))

#define RKNPU_MAX_CORES 3
#define RKNPU_CORE0_MASK 0x1
#define RKNPU_CORE1_MASK 0x2
#define RKNPU_CORE2_MASK 0x4
#define RKNPU_CORE_AUTO_MASK 0x0

#define RKNPU_OFFSET_VERSION 0x0000
#define RKNPU_OFFSET_VERSION_NUM 0x0004
#define RKNPU_OFFSET_PC_OP_EN 0x0008
#define RKNPU_OFFSET_PC_DATA_ADDR 0x0010
#define RKNPU_OFFSET_PC_DATA_AMOUNT 0x0014
#define RKNPU_OFFSET_INT_MASK 0x0020
#define RKNPU_OFFSET_INT_CLEAR 0x0024
#define RKNPU_OFFSET_INT_STATUS 0x0028
#define RKNPU_OFFSET_INT_RAW_STATUS 0x002c
#define RKNPU_OFFSET_PC_TASK_CONTROL 0x0030
#define RKNPU_OFFSET_PC_DMA_BASE_ADDR 0x0034
#define RKNPU_OFFSET_PC_TASK_STATUS 0x003c

#define RKNPU_OFFSET_CNA_S_POINTER 0x1004
#define RKNPU_OFFSET_CORE_S_POINTER 0x3004

#define RKNPU_OFFSET_CLR_ALL_RW_AMOUNT 0x8010
#define RKNPU_OFFSET_DT_WR_AMOUNT 0x8034
#define RKNPU_OFFSET_DT_RD_AMOUNT 0x8038
#define RKNPU_OFFSET_WT_RD_AMOUNT 0x803c

#define RKNPU_INT_CLEAR 0x1ffff
#define RKNPU_PC_DATA_EXTRA_AMOUNT 4
#define RK3588_PC_DATA_AMOUNT_SCALE 2
#define RK3588_PC_TASK_NUMBER_BITS 12
#define RK3588_PC_TASK_NUMBER_MASK 0xfff
#define RK3588_MAX_SUBMIT_NUMBER ((1 << RK3588_PC_TASK_NUMBER_BITS) - 1)
#define RK3588_NPU_DEFAULT_RATE_HZ 750000000UL

enum e_rknpu_mem_sync_mode {
	RKNPU_MEM_SYNC_TO_DEVICE = 1 << 0,
	RKNPU_MEM_SYNC_FROM_DEVICE = 1 << 1,
};

enum e_rknpu_job_mode {
	RKNPU_JOB_SLAVE = 0 << 0,
	RKNPU_JOB_PC = 1 << 0,
	RKNPU_JOB_BLOCK = 0 << 1,
	RKNPU_JOB_NONBLOCK = 1 << 1,
	RKNPU_JOB_PINGPONG = 1 << 2,
	RKNPU_JOB_FENCE_IN = 1 << 3,
	RKNPU_JOB_FENCE_OUT = 1 << 4,
};

enum e_rknpu_action {
	RKNPU_GET_HW_VERSION = 0,
	RKNPU_GET_DRV_VERSION = 1,
	RKNPU_GET_FREQ = 2,
	RKNPU_SET_FREQ = 3,
	RKNPU_GET_VOLT = 4,
	RKNPU_SET_VOLT = 5,
	RKNPU_ACT_RESET = 6,
	RKNPU_GET_BW_PRIORITY = 7,
	RKNPU_SET_BW_PRIORITY = 8,
	RKNPU_GET_BW_EXPECT = 9,
	RKNPU_SET_BW_EXPECT = 10,
	RKNPU_GET_BW_TW = 11,
	RKNPU_SET_BW_TW = 12,
	RKNPU_ACT_CLR_TOTAL_RW_AMOUNT = 13,
	RKNPU_GET_DT_WR_AMOUNT = 14,
	RKNPU_GET_DT_RD_AMOUNT = 15,
	RKNPU_GET_WT_RD_AMOUNT = 16,
	RKNPU_GET_TOTAL_RW_AMOUNT = 17,
	RKNPU_GET_IOMMU_EN = 18,
	RKNPU_SET_PROC_NICE = 19,
	RKNPU_POWER_ON = 20,
	RKNPU_POWER_OFF = 21,
	RKNPU_GET_TOTAL_SRAM_SIZE = 22,
	RKNPU_GET_FREE_SRAM_SIZE = 23,
};

struct rknpu_mem_create {
	__u32 handle;
	__u32 flags;
	__u64 size;
	__u64 obj_addr;
	__u64 dma_addr;
	__u64 sram_size;
};

struct rknpu_mem_map {
	__u32 handle;
	__u32 reserved;
	__u64 offset;
};

struct rknpu_mem_destroy {
	__u32 handle;
	__u32 reserved;
	__u64 obj_addr;
};

struct rknpu_mem_sync {
	__u32 flags;
	__u32 reserved;
	__u64 obj_addr;
	__u64 offset;
	__u64 size;
};

struct rknpu_task {
	__u32 flags;
	__u32 op_idx;
	__u32 enable_mask;
	__u32 int_mask;
	__u32 int_clear;
	__u32 int_status;
	__u32 regcfg_amount;
	__u32 regcfg_offset;
	__u64 regcmd_addr;
} __packed;

struct rknpu_subcore_task {
	__u32 task_start;
	__u32 task_number;
};

struct rknpu_submit {
	__u32 flags;
	__u32 timeout;
	__u32 task_start;
	__u32 task_number;
	__u32 task_counter;
	__s32 priority;
	__u64 task_obj_addr;
	__u64 regcfg_obj_addr;
	__u64 task_base_addr;
	__u64 user_data;
	__u32 core_mask;
	__s32 fence_fd;
	struct rknpu_subcore_task subcore_task[5];
};

struct rknpu_action {
	__u32 flags;
	__u32 value;
};

#define RKNPU_ACTION 0x00
#define RKNPU_SUBMIT 0x01
#define RKNPU_MEM_CREATE 0x02
#define RKNPU_MEM_MAP 0x03
#define RKNPU_MEM_DESTROY 0x04
#define RKNPU_MEM_SYNC 0x05

#define RKNPU_IOC_MAGIC 'r'
#define IOCTL_RKNPU_ACTION _IOWR(RKNPU_IOC_MAGIC, RKNPU_ACTION, struct rknpu_action)
#define IOCTL_RKNPU_SUBMIT _IOWR(RKNPU_IOC_MAGIC, RKNPU_SUBMIT, struct rknpu_submit)
#define IOCTL_RKNPU_MEM_CREATE _IOWR(RKNPU_IOC_MAGIC, RKNPU_MEM_CREATE, struct rknpu_mem_create)
#define IOCTL_RKNPU_MEM_MAP _IOWR(RKNPU_IOC_MAGIC, RKNPU_MEM_MAP, struct rknpu_mem_map)
#define IOCTL_RKNPU_MEM_DESTROY _IOWR(RKNPU_IOC_MAGIC, RKNPU_MEM_DESTROY, struct rknpu_mem_destroy)
#define IOCTL_RKNPU_MEM_SYNC _IOWR(RKNPU_IOC_MAGIC, RKNPU_MEM_SYNC, struct rknpu_mem_sync)

#define DRM_IOCTL_RKNPU_ACTION DRM_IOWR(DRM_COMMAND_BASE + RKNPU_ACTION, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT DRM_IOWR(DRM_COMMAND_BASE + RKNPU_SUBMIT, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_CREATE, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_MAP, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_DESTROY, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC DRM_IOWR(DRM_COMMAND_BASE + RKNPU_MEM_SYNC, struct rknpu_mem_sync)

struct rknpu_mem_object {
	struct kref refcount;
	struct list_head head;
	void *kv_addr;
	phys_addr_t phys_addr;
	dma_addr_t dma_addr;
	size_t size;
	bool in_session;
};

struct rknpu_session {
	struct mutex lock;
	struct list_head list;
};

struct rknpu_core_dev {
	int index;
	struct device *dev;
	void __iomem *base;
	int irq;
	struct clk_bulk_data *clks;
	int num_clks;
	struct clk *npu_clk;
	struct reset_control *srst_a;
	struct reset_control *srst_h;
	spinlock_t irq_lock;
	struct mutex lock;
	wait_queue_head_t wq;
	bool job_active;
	bool job_done;
	u32 expected_int_mask;
	u32 int_status;
	u64 job_start_ns;
	bool ready;
};

struct rknpu_global {
	struct mutex submit_lock;
	atomic_t rr_core;
	struct rknpu_core_dev cores[RKNPU_MAX_CORES];
	struct platform_device *dyn_pdevs[RKNPU_MAX_CORES];
	struct miscdevice miscdev;
	bool misc_registered;
};

static struct rknpu_global g_rknpu;

static unsigned long hw_busy_ns;
module_param(hw_busy_ns, ulong, 0644);
MODULE_PARM_DESC(hw_busy_ns, "Cumulative hardware NPU busy time in nanoseconds");

static unsigned long hw_jobs;
module_param(hw_jobs, ulong, 0644);
MODULE_PARM_DESC(hw_jobs, "Cumulative hardware NPU jobs completed");

static unsigned long npu_freq_hz = RK3588_NPU_DEFAULT_RATE_HZ;
module_param(npu_freq_hz, ulong, 0444);
MODULE_PARM_DESC(npu_freq_hz, "Current RK3588 NPU clock rate in Hz");

static inline u32 rknpu_fuzz_status(u32 status)
{
	u32 fuzz_status = 0;

	if ((status & 0x3) != 0)
		fuzz_status |= 0x3;
	if ((status & 0xc) != 0)
		fuzz_status |= 0xc;
	if ((status & 0x30) != 0)
		fuzz_status |= 0x30;
	if ((status & 0xc0) != 0)
		fuzz_status |= 0xc0;
	if ((status & 0x300) != 0)
		fuzz_status |= 0x300;
	if ((status & 0xc00) != 0)
		fuzz_status |= 0xc00;

	return fuzz_status;
}

static irqreturn_t rknpu_core_irq_handler(int irq, void *data)
{
	struct rknpu_core_dev *core = data;
	void __iomem *base = core->base;
	unsigned long flags;
	u32 status;
	u64 now_ns;

	status = readl(base + RKNPU_OFFSET_INT_STATUS);
	if (!status)
		return IRQ_NONE;

	now_ns = ktime_get_ns();

	spin_lock_irqsave(&core->irq_lock, flags);
	writel(RKNPU_INT_CLEAR, base + RKNPU_OFFSET_INT_CLEAR);

	if (core->job_active) {
		core->int_status = status;
		if (rknpu_fuzz_status(status) == core->expected_int_mask ||
		    (rknpu_fuzz_status(status) & core->expected_int_mask) ==
			    core->expected_int_mask) {
			core->job_active = false;
			core->job_done = true;
			if (now_ns > core->job_start_ns)
				WRITE_ONCE(hw_busy_ns,
					   READ_ONCE(hw_busy_ns) +
						   (unsigned long)(now_ns - core->job_start_ns));
			WRITE_ONCE(hw_jobs, READ_ONCE(hw_jobs) + 1);
			wake_up(&core->wq);
		}
	}
	spin_unlock_irqrestore(&core->irq_lock, flags);

	return IRQ_HANDLED;
}

static int rknpu_soft_reset(void)
{
	int i;

	for (i = 0; i < RKNPU_MAX_CORES; i++) {
		struct rknpu_core_dev *core = &g_rknpu.cores[i];

		if (!core->ready)
			continue;
		if (core->srst_a)
			reset_control_assert(core->srst_a);
		if (core->srst_h)
			reset_control_assert(core->srst_h);
		udelay(10);
		if (core->srst_a)
			reset_control_deassert(core->srst_a);
		if (core->srst_h)
			reset_control_deassert(core->srst_h);
	}
	return 0;
}

static void rknpu_mem_obj_release(struct kref *kref)
{
	struct rknpu_mem_object *obj =
		container_of(kref, struct rknpu_mem_object, refcount);

	if (obj->kv_addr) {
		free_pages_exact(obj->kv_addr, obj->size);
		obj->kv_addr = NULL;
	}
	kfree(obj);
}

static int rknpu_buf_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct rknpu_mem_object *obj = filp->private_data;
	unsigned long vm_size;

	if (!obj || !obj->kv_addr)
		return -EINVAL;

	vm_size = vma->vm_end - vma->vm_start;
	if (vm_size > obj->size)
		return -EINVAL;

	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);
	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);

	return remap_pfn_range(vma, vma->vm_start,
			       obj->phys_addr >> PAGE_SHIFT, vm_size,
			       vma->vm_page_prot);
}

static int rknpu_buf_release(struct inode *inode, struct file *filp)
{
	struct rknpu_mem_object *obj = filp->private_data;

	if (obj) {
		filp->private_data = NULL;
		kref_put(&obj->refcount, rknpu_mem_obj_release);
	}
	return 0;
}

static const struct file_operations rknpu_buf_fops = {
	.owner = THIS_MODULE,
	.mmap = rknpu_buf_mmap,
	.release = rknpu_buf_release,
	.llseek = noop_llseek,
};

static struct rknpu_mem_object *
rknpu_session_find_obj(struct rknpu_session *session, __u64 obj_addr)
{
	struct rknpu_mem_object *entry;

	list_for_each_entry(entry, &session->list, head) {
		if ((__u64)(uintptr_t)entry == obj_addr) {
			kref_get(&entry->refcount);
			return entry;
		}
	}
	return NULL;
}

static int rknpu_mem_create_ioctl(struct rknpu_session *session,
				  unsigned long data)
{
	struct rknpu_mem_create args;
	struct rknpu_mem_object *obj = NULL;
	struct device *dma_dev = g_rknpu.cores[0].dev;
	size_t alloc_size;
	void *kv_addr;
	phys_addr_t phys;
	int fd;

	if (copy_from_user(&args, (void __user *)data, sizeof(args)))
		return -EFAULT;

	if (args.size == 0 || args.size > (512ULL << 20))
		return -EINVAL;

	alloc_size = PAGE_ALIGN((size_t)args.size);
	if (alloc_size == 0)
		alloc_size = PAGE_SIZE;

	obj = kzalloc(sizeof(*obj), GFP_KERNEL);
	if (!obj)
		return -ENOMEM;

	/* Allocate physically contiguous pages in <4GB ZONE_DMA for 32-bit NPU DMA */
	kv_addr = alloc_pages_exact(alloc_size,
				    GFP_KERNEL | __GFP_DMA | __GFP_ZERO);
	if (!kv_addr)
		kv_addr = alloc_pages_exact(alloc_size, GFP_KERNEL | __GFP_ZERO);
	if (!kv_addr) {
		kfree(obj);
		return -ENOMEM;
	}

	phys = virt_to_phys(kv_addr);
	if (phys + alloc_size > 0x100000000ULL) {
		pr_warn("rknpu: allocated buffer above 4GB (%pa), NPU DMA32 may truncate\n",
			&phys);
	}

	kref_init(&obj->refcount);
	INIT_LIST_HEAD(&obj->head);
	obj->kv_addr = kv_addr;
	obj->phys_addr = phys;
	obj->dma_addr = (dma_addr_t)phys;
	obj->size = alloc_size;

	if (dma_dev)
		dma_sync_single_for_device(dma_dev, obj->dma_addr, obj->size,
					   DMA_BIDIRECTIONAL);

	fd = get_unused_fd_flags(O_CLOEXEC | O_RDWR);
	if (fd < 0) {
		kref_put(&obj->refcount, rknpu_mem_obj_release);
		return fd;
	}

	/* Give one reference to the anon_inode file */
	kref_get(&obj->refcount);
	{
		struct file *file = anon_inode_getfile("rknpu_mem",
						       &rknpu_buf_fops, obj,
						       O_CLOEXEC | O_RDWR);
		if (IS_ERR(file)) {
			put_unused_fd(fd);
			kref_put(&obj->refcount, rknpu_mem_obj_release);
			kref_put(&obj->refcount, rknpu_mem_obj_release);
			return PTR_ERR(file);
		}

		args.size = obj->size;
		args.obj_addr = (__u64)(uintptr_t)obj;
		args.dma_addr = (__u64)obj->dma_addr;
		args.handle = (__u32)fd;

		if (copy_to_user((void __user *)data, &args, sizeof(args))) {
			put_unused_fd(fd);
			fput(file);
			kref_put(&obj->refcount, rknpu_mem_obj_release);
			return -EFAULT;
		}

		mutex_lock(&session->lock);
		obj->in_session = true;
		list_add_tail(&obj->head, &session->list);
		mutex_unlock(&session->lock);

		fd_install(fd, file);
	}

	return 0;
}

static int rknpu_mem_destroy_ioctl(struct rknpu_session *session,
				   unsigned long data)
{
	struct rknpu_mem_destroy args;
	struct rknpu_mem_object *entry, *tmp, *found = NULL;

	if (copy_from_user(&args, (void __user *)data, sizeof(args)))
		return -EFAULT;

	mutex_lock(&session->lock);
	list_for_each_entry_safe(entry, tmp, &session->list, head) {
		if ((__u64)(uintptr_t)entry == args.obj_addr) {
			list_del_init(&entry->head);
			entry->in_session = false;
			found = entry;
			break;
		}
	}
	mutex_unlock(&session->lock);

	if (found)
		kref_put(&found->refcount, rknpu_mem_obj_release);

	return 0;
}

static int rknpu_mem_sync_ioctl(struct rknpu_session *session,
				unsigned long data)
{
	struct rknpu_mem_sync args;
	struct rknpu_mem_object *obj;
	struct device *dma_dev = g_rknpu.cores[0].dev;
	size_t sync_offset, sync_size;

	if (copy_from_user(&args, (void __user *)data, sizeof(args)))
		return -EFAULT;

	mutex_lock(&session->lock);
	obj = rknpu_session_find_obj(session, args.obj_addr);
	mutex_unlock(&session->lock);

	if (!obj)
		return 0;

	if (dma_dev && args.offset < obj->size) {
		sync_offset = (size_t)args.offset;
		sync_size = (size_t)args.size;
		if (sync_size == 0 || sync_offset + sync_size > obj->size)
			sync_size = obj->size - sync_offset;

		if (args.flags & RKNPU_MEM_SYNC_TO_DEVICE)
			dma_sync_single_for_device(dma_dev,
						   obj->dma_addr + sync_offset,
						   sync_size, DMA_TO_DEVICE);
		if (args.flags & RKNPU_MEM_SYNC_FROM_DEVICE)
			dma_sync_single_for_cpu(dma_dev,
						obj->dma_addr + sync_offset,
						sync_size, DMA_FROM_DEVICE);
	}

	kref_put(&obj->refcount, rknpu_mem_obj_release);
	return 0;
}

static int rknpu_action_ioctl(unsigned long data)
{
	struct rknpu_action args;
	struct rknpu_core_dev *core0 = &g_rknpu.cores[0];
	void __iomem *base0 = core0->base;
	int ret = 0;

	if (copy_from_user(&args, (void __user *)data, sizeof(args)))
		return -EFAULT;

	switch (args.flags) {
	case RKNPU_GET_HW_VERSION:
		if (!base0)
			return -ENODEV;
		args.value = readl(base0 + RKNPU_OFFSET_VERSION) +
			     (readl(base0 + RKNPU_OFFSET_VERSION_NUM) & 0xffff);
		break;
	case RKNPU_GET_DRV_VERSION:
		args.value = RKNPU_DRV_VERSION_CODE;
		break;
	case RKNPU_GET_FREQ:
		if (core0->npu_clk)
			args.value = (__u32)clk_get_rate(core0->npu_clk);
		else
			args.value = (__u32)READ_ONCE(npu_freq_hz);
		break;
	case RKNPU_SET_FREQ:
	case RKNPU_GET_VOLT:
	case RKNPU_SET_VOLT:
	case RKNPU_GET_BW_PRIORITY:
	case RKNPU_SET_BW_PRIORITY:
	case RKNPU_GET_BW_EXPECT:
	case RKNPU_SET_BW_EXPECT:
	case RKNPU_GET_BW_TW:
	case RKNPU_SET_BW_TW:
	case RKNPU_ACT_CLR_TOTAL_RW_AMOUNT:
	case RKNPU_GET_DT_WR_AMOUNT:
	case RKNPU_GET_DT_RD_AMOUNT:
	case RKNPU_GET_WT_RD_AMOUNT:
	case RKNPU_GET_TOTAL_RW_AMOUNT:
		args.value = 0;
		break;
	case RKNPU_ACT_RESET:
		ret = rknpu_soft_reset();
		break;
	case RKNPU_GET_IOMMU_EN:
		/* Non-IOMMU mode: contiguous physical DMA32 buffers */
		args.value = 0;
		break;
	case RKNPU_SET_PROC_NICE: {
		s32 nice_val = *(s32 *)&args.value;

		if (nice_val < MIN_NICE)
			nice_val = MIN_NICE;
		if (nice_val > MAX_NICE)
			nice_val = MAX_NICE;
		set_user_nice(current, nice_val);
		break;
	}
	case RKNPU_POWER_ON:
	case RKNPU_POWER_OFF:
		args.value = 0;
		break;
	case RKNPU_GET_TOTAL_SRAM_SIZE:
	case RKNPU_GET_FREE_SRAM_SIZE:
		args.value = 0;
		break;
	default:
		args.value = 0;
		break;
	}

	if (copy_to_user((void __user *)data, &args, sizeof(args)))
		return -EFAULT;

	return ret;
}

static int rknpu_commit_core_pc(struct rknpu_core_dev *core,
				struct rknpu_submit *args,
				struct rknpu_task *task_base,
				size_t max_tasks,
				int task_start,
				int task_number,
				struct rknpu_task **out_last_task)
{
	void __iomem *base = core->base;
	struct rknpu_task *first_task;
	struct rknpu_task *last_task;
	int task_end;
	int task_pp_en = (args->flags & RKNPU_JOB_PINGPONG) ? 1 : 0;
	u32 amount;
	unsigned long flags;
	int i;

	if (task_number <= 0)
		return -EINVAL;
	if (task_number > RK3588_MAX_SUBMIT_NUMBER)
		task_number = RK3588_MAX_SUBMIT_NUMBER;

	task_end = task_start + task_number - 1;
	if (task_start < 0 || (size_t)task_end >= max_tasks)
		return -EINVAL;

	first_task = &task_base[task_start];
	last_task = &task_base[task_end];
	*out_last_task = last_task;

	/* Switch PC to slave mode */
	writel(0x1, base + RKNPU_OFFSET_PC_DATA_ADDR);

	for (i = 0; i < RKNPU_MAX_CORES; i++) {
		if (i == core->index) {
			writel(0xe + 0x10000000 * i,
			       base + RKNPU_OFFSET_CNA_S_POINTER);
			writel(0xe + 0x10000000 * i,
			       base + RKNPU_OFFSET_CORE_S_POINTER);
		}
	}

	writel((u32)first_task->regcmd_addr, base + RKNPU_OFFSET_PC_DATA_ADDR);

	amount = (first_task->regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT +
		  RK3588_PC_DATA_AMOUNT_SCALE - 1) /
			 RK3588_PC_DATA_AMOUNT_SCALE -
		 1;
	writel(amount, base + RKNPU_OFFSET_PC_DATA_AMOUNT);
	writel(last_task->int_mask, base + RKNPU_OFFSET_INT_MASK);
	writel(first_task->int_mask, base + RKNPU_OFFSET_INT_CLEAR);
	writel(((0x6 | task_pp_en) << RK3588_PC_TASK_NUMBER_BITS) | task_number,
	       base + RKNPU_OFFSET_PC_TASK_CONTROL);
	writel((u32)args->task_base_addr, base + RKNPU_OFFSET_PC_DMA_BASE_ADDR);

	spin_lock_irqsave(&core->irq_lock, flags);
	core->expected_int_mask = last_task->int_mask;
	core->int_status = 0;
	core->job_done = false;
	core->job_active = true;
	core->job_start_ns = ktime_get_ns();
	spin_unlock_irqrestore(&core->irq_lock, flags);

	/* Ensure all DMA buffer writes and PC registers are visible before OP_EN */
	wmb();
	writel(0x1, base + RKNPU_OFFSET_PC_OP_EN);
	writel(0x0, base + RKNPU_OFFSET_PC_OP_EN);

	return task_number;
}

static int rknpu_submit_ioctl(struct rknpu_session *session, unsigned long data)
{
	struct rknpu_submit args;
	struct rknpu_mem_object *task_obj;
	struct rknpu_task *task_base;
	size_t max_tasks;
	u32 core_mask;
	int use_core_num;
	unsigned long timeout_jiffies;
	int ret = 0;
	int c;

	if (copy_from_user(&args, (void __user *)data, sizeof(args)))
		return -EFAULT;

	if (args.task_number == 0 || !(args.flags & RKNPU_JOB_PC))
		return -EINVAL;

	mutex_lock(&session->lock);
	task_obj = rknpu_session_find_obj(session, args.task_obj_addr);
	mutex_unlock(&session->lock);

	if (!task_obj || !task_obj->kv_addr) {
		if (task_obj)
			kref_put(&task_obj->refcount, rknpu_mem_obj_release);
		return -EINVAL;
	}

	task_base = (struct rknpu_task *)task_obj->kv_addr;
	max_tasks = task_obj->size / sizeof(struct rknpu_task);

	/* Ensure write-combined userspace task descriptors are visible */
	mb();

	core_mask = args.core_mask & 0x7;
	if (core_mask == RKNPU_CORE_AUTO_MASK) {
		unsigned int rr =
			(unsigned int)atomic_inc_return(&g_rknpu.rr_core) %
			RKNPU_MAX_CORES;
		core_mask = 1U << rr;
	}

	/* Fall back to core 0 if any requested core is not available */
	for (c = 0; c < RKNPU_MAX_CORES; c++) {
		if ((core_mask & (1U << c)) && !g_rknpu.cores[c].ready)
			core_mask = RKNPU_CORE0_MASK;
	}
	args.core_mask = core_mask;

	use_core_num = ((core_mask & RKNPU_CORE0_MASK) ? 1 : 0) +
		       ((core_mask & RKNPU_CORE1_MASK) ? 1 : 0) +
		       ((core_mask & RKNPU_CORE2_MASK) ? 1 : 0);

	timeout_jiffies = msecs_to_jiffies(args.timeout ? args.timeout : 5000);

	if (use_core_num == 1) {
		int core_idx = (core_mask & RKNPU_CORE0_MASK) ? 0 :
			       (core_mask & RKNPU_CORE1_MASK) ? 1 : 2;
		struct rknpu_core_dev *core = &g_rknpu.cores[core_idx];
		int t_start = args.subcore_task[core_idx].task_start;
		int t_num = args.subcore_task[core_idx].task_number;
		int submitted = 0;

		if (t_num == 0 && args.subcore_task[0].task_number > 0) {
			t_start = args.subcore_task[0].task_start;
			t_num = args.subcore_task[0].task_number;
		}
		if (t_num == 0) {
			t_start = args.task_start;
			t_num = args.task_number;
		}

		mutex_lock(&core->lock);
		while (submitted < t_num) {
			struct rknpu_task *last_task = NULL;
			int chunk = rknpu_commit_core_pc(core, &args, task_base,
							 max_tasks,
							 t_start + submitted,
							 t_num - submitted,
							 &last_task);
			long wait_ret;

			if (chunk < 0) {
				ret = chunk;
				break;
			}

			wait_ret = wait_event_timeout(core->wq, core->job_done,
						      timeout_jiffies);
			if (wait_ret <= 0 || !core->job_done) {
				u32 task_status = readl(
					core->base + RKNPU_OFFSET_PC_TASK_STATUS);
				args.task_counter =
					task_status & RK3588_PC_TASK_NUMBER_MASK;
				pr_err("rknpu: core %d timeout, int_status=%#x raw=%#x task_status=%#x\n",
				       core_idx,
				       readl(core->base + RKNPU_OFFSET_INT_STATUS),
				       readl(core->base + RKNPU_OFFSET_INT_RAW_STATUS),
				       task_status);
				rknpu_soft_reset();
				ret = -ETIMEDOUT;
				break;
			}

			if (last_task)
				last_task->int_status = core->int_status;
			submitted += chunk;
		}
		mutex_unlock(&core->lock);

		if (ret == 0)
			args.task_counter = args.task_number;
	} else {
		/* Multi-core collaborative submission (2 or 3 cores) */
		struct rknpu_task *last_tasks[RKNPU_MAX_CORES] = { NULL };
		bool active_cores[RKNPU_MAX_CORES] = { false };

		for (c = 0; c < RKNPU_MAX_CORES; c++) {
			if (core_mask & (1U << c))
				mutex_lock(&g_rknpu.cores[c].lock);
		}

		for (c = 0; c < RKNPU_MAX_CORES; c++) {
			int sub_idx = (use_core_num == 3) ? (c + 2) : c;
			int t_start = args.subcore_task[sub_idx].task_start;
			int t_num = args.subcore_task[sub_idx].task_number;
			int chunk;

			if (!(core_mask & (1U << c)) || t_num == 0)
				continue;

			chunk = rknpu_commit_core_pc(&g_rknpu.cores[c], &args,
						     task_base, max_tasks,
						     t_start, t_num,
						     &last_tasks[c]);
			if (chunk < 0) {
				ret = chunk;
				break;
			}
			active_cores[c] = true;
		}

		for (c = 0; c < RKNPU_MAX_CORES && ret == 0; c++) {
			struct rknpu_core_dev *core = &g_rknpu.cores[c];
			long wait_ret;

			if (!active_cores[c])
				continue;

			wait_ret = wait_event_timeout(core->wq, core->job_done,
						      timeout_jiffies);
			if (wait_ret <= 0 || !core->job_done) {
				pr_err("rknpu: multi-core %d timeout\n", c);
				rknpu_soft_reset();
				ret = -ETIMEDOUT;
				break;
			}
			if (last_tasks[c])
				last_tasks[c]->int_status = core->int_status;
		}

		for (c = RKNPU_MAX_CORES - 1; c >= 0; c--) {
			if (core_mask & (1U << c))
				mutex_unlock(&g_rknpu.cores[c].lock);
		}

		if (ret == 0)
			args.task_counter = args.task_number;
	}

	/* Ensure NPU DMA writes and last_task->int_status are visible */
	mb();
	kref_put(&task_obj->refcount, rknpu_mem_obj_release);

	if (copy_to_user((void __user *)data, &args, sizeof(args)))
		return -EFAULT;

	return ret;
}

static int rknpu_open(struct inode *inode, struct file *file)
{
	struct rknpu_session *session;

	session = kzalloc(sizeof(*session), GFP_KERNEL);
	if (!session)
		return -ENOMEM;

	mutex_init(&session->lock);
	INIT_LIST_HEAD(&session->list);
	file->private_data = session;

	return nonseekable_open(inode, file);
}

static int rknpu_release(struct inode *inode, struct file *file)
{
	struct rknpu_session *session = file->private_data;
	struct rknpu_mem_object *entry, *tmp;
	LIST_HEAD(local_list);

	if (!session)
		return 0;

	mutex_lock(&session->lock);
	list_replace_init(&session->list, &local_list);
	file->private_data = NULL;
	mutex_unlock(&session->lock);

	list_for_each_entry_safe(entry, tmp, &local_list, head) {
		list_del_init(&entry->head);
		entry->in_session = false;
		kref_put(&entry->refcount, rknpu_mem_obj_release);
	}

	kfree(session);
	return 0;
}

static long rknpu_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct rknpu_session *session = file->private_data;

	if (!session)
		return -EINVAL;

	switch (cmd) {
	case IOCTL_RKNPU_ACTION:
	case DRM_IOCTL_RKNPU_ACTION:
		return rknpu_action_ioctl(arg);
	case IOCTL_RKNPU_SUBMIT:
	case DRM_IOCTL_RKNPU_SUBMIT:
		return rknpu_submit_ioctl(session, arg);
	case IOCTL_RKNPU_MEM_CREATE:
	case DRM_IOCTL_RKNPU_MEM_CREATE:
		return rknpu_mem_create_ioctl(session, arg);
	case IOCTL_RKNPU_MEM_MAP:
	case DRM_IOCTL_RKNPU_MEM_MAP:
		return 0;
	case IOCTL_RKNPU_MEM_DESTROY:
	case DRM_IOCTL_RKNPU_MEM_DESTROY:
		return rknpu_mem_destroy_ioctl(session, arg);
	case IOCTL_RKNPU_MEM_SYNC:
	case DRM_IOCTL_RKNPU_MEM_SYNC:
		return rknpu_mem_sync_ioctl(session, arg);
	default:
		return -EINVAL;
	}
}

static const struct file_operations rknpu_fops = {
	.owner = THIS_MODULE,
	.open = rknpu_open,
	.release = rknpu_release,
	.unlocked_ioctl = rknpu_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = rknpu_ioctl,
#endif
	.llseek = noop_llseek,
};

static int rknpu_core_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct resource *res;
	struct rknpu_core_dev *core;
	int core_idx = -1;
	int ret;
	u32 hw_ver;

	res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "pc");
	if (!res)
		res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!res)
		return -ENODEV;

	switch ((u32)res->start) {
	case 0xfdab0000:
		core_idx = 0;
		break;
	case 0xfdac0000:
		core_idx = 1;
		break;
	case 0xfdad0000:
		core_idx = 2;
		break;
	default:
		dev_err(dev, "unknown NPU core base address %pa\n", &res->start);
		return -EINVAL;
	}

	core = &g_rknpu.cores[core_idx];
	core->index = core_idx;
	core->dev = dev;
	spin_lock_init(&core->irq_lock);
	init_waitqueue_head(&core->wq);

	/* Map full 64KB core register window covering PC, CNA, CORE, DMA */
	core->base = devm_ioremap(dev, res->start, 0x10000);
	if (!core->base)
		return -ENOMEM;

	core->srst_a = devm_reset_control_get_optional_exclusive(dev, "srst_a");
	if (IS_ERR(core->srst_a))
		return PTR_ERR(core->srst_a);

	core->srst_h = devm_reset_control_get_optional_exclusive(dev, "srst_h");
	if (IS_ERR(core->srst_h))
		return PTR_ERR(core->srst_h);

	core->num_clks = devm_clk_bulk_get_all(dev, &core->clks);
	if (core->num_clks < 0)
		return core->num_clks;

	core->npu_clk = devm_clk_get_optional(dev, "npu");
	if (IS_ERR(core->npu_clk))
		core->npu_clk = NULL;
	if (!core->npu_clk && core->num_clks >= 3)
		core->npu_clk = core->clks[2].clk;

	dma_set_max_seg_size(dev, UINT_MAX);
	ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(32));
	if (ret)
		ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(40));

	pm_runtime_enable(dev);
	ret = pm_runtime_resume_and_get(dev);
	if (ret < 0) {
		pm_runtime_disable(dev);
		dev_err(dev, "failed to resume power domain for core %d: %d\n",
			core_idx, ret);
		return ret;
	}

	ret = clk_bulk_prepare_enable(core->num_clks, core->clks);
	if (ret) {
		pm_runtime_put_sync(dev);
		pm_runtime_disable(dev);
		return ret;
	}

	/*
	 * rk3588-base.dtsi sets assigned-clock-rates = <200000000> on scmi_clk_npu.
	 * Raise it to 800 MHz (or highest accepted discrete SCMI OPP) so the NPU
	 * runs at full operating frequency within boot voltage.
	 */
	if (core->npu_clk) {
		static const unsigned long target_rates[] = {
			800000000UL, 750000000UL, 700000000UL, 600000000UL
		};
		unsigned long cur = clk_get_rate(core->npu_clk);
		size_t r;

		for (r = 0; r < ARRAY_SIZE(target_rates); r++) {
			if (cur >= target_rates[r])
				break;
			if (clk_set_rate(core->npu_clk, target_rates[r]) == 0) {
				cur = clk_get_rate(core->npu_clk);
				if (cur >= target_rates[r])
					break;
			}
		}
		WRITE_ONCE(npu_freq_hz, cur);
	}

	if (core->srst_a)
		reset_control_deassert(core->srst_a);
	if (core->srst_h)
		reset_control_deassert(core->srst_h);

	core->irq = platform_get_irq(pdev, 0);
	if (core->irq < 0) {
		clk_bulk_disable_unprepare(core->num_clks, core->clks);
		pm_runtime_put_sync(dev);
		pm_runtime_disable(dev);
		return core->irq;
	}

	ret = devm_request_irq(dev, core->irq, rknpu_core_irq_handler,
			       IRQF_SHARED, dev_name(dev), core);
	if (ret) {
		clk_bulk_disable_unprepare(core->num_clks, core->clks);
		pm_runtime_put_sync(dev);
		pm_runtime_disable(dev);
		return ret;
	}

	hw_ver = readl(core->base + RKNPU_OFFSET_VERSION) +
		 (readl(core->base + RKNPU_OFFSET_VERSION_NUM) & 0xffff);
	core->ready = true;
	platform_set_drvdata(pdev, core);

	dev_info(dev, "RK3588 RKNPU core %d ready (hw_ver=%#x, clk=%lu Hz, irq=%d)\n",
		 core_idx, hw_ver, READ_ONCE(npu_freq_hz), core->irq);

	if (core_idx == 0 && !g_rknpu.misc_registered) {
		g_rknpu.miscdev.minor = MISC_DYNAMIC_MINOR;
		g_rknpu.miscdev.name = DRIVER_NAME;
		g_rknpu.miscdev.fops = &rknpu_fops;
		g_rknpu.miscdev.mode = 0666;
		ret = misc_register(&g_rknpu.miscdev);
		if (ret == 0) {
			g_rknpu.misc_registered = true;
			dev_info(dev, "registered /dev/rknpu (drv_ver=%d)\n",
				 RKNPU_DRV_VERSION_CODE);
		} else {
			dev_err(dev, "failed to register /dev/rknpu: %d\n", ret);
		}
	}

	return 0;
}

static void rknpu_core_remove(struct platform_device *pdev)
{
	struct rknpu_core_dev *core = platform_get_drvdata(pdev);

	if (!core)
		return;

	core->ready = false;
	clk_bulk_disable_unprepare(core->num_clks, core->clks);
	pm_runtime_put_sync(&pdev->dev);
	pm_runtime_disable(&pdev->dev);
}

static const struct of_device_id rknpu_core_dt_match[] = {
	{ .compatible = "rockchip,rk3588-rknn-core" },
	{},
};
MODULE_DEVICE_TABLE(of, rknpu_core_dt_match);

static struct platform_driver rknpu_core_driver = {
	.probe = rknpu_core_probe,
	.remove = rknpu_core_remove,
	.driver = {
		.name = "rk3588-rknpu-core",
		.of_match_table = rknpu_core_dt_match,
	},
};

static const char *const rk3588_npu_dt_paths[RKNPU_MAX_CORES] = {
	"/npu@fdab0000",
	"/npu@fdac0000",
	"/npu@fdad0000",
};

static int __init rknpu_module_init(void)
{
	int i;

	mutex_init(&g_rknpu.submit_lock);
	atomic_set(&g_rknpu.rr_core, 0);
	for (i = 0; i < RKNPU_MAX_CORES; i++)
		mutex_init(&g_rknpu.cores[i].lock);

	/*
	 * Ensure the 3 NPU core DT nodes are enabled and populated as
	 * platform_devices even if the currently booted DTB had status="disabled".
	 */
	for (i = 0; i < RKNPU_MAX_CORES; i++) {
		struct device_node *np =
			of_find_node_by_path(rk3588_npu_dt_paths[i]);
		struct platform_device * exist_pdev;

		if (!np)
			continue;

		if (!of_device_is_available(np)) {
			struct property *prop =
				of_find_property(np, "status", NULL);
			if (prop) {
				pr_info("rknpu: enabling DT node %s\n",
					rk3588_npu_dt_paths[i]);
				of_remove_property(np, prop);
			}
		}

		exist_pdev = of_find_device_by_node(np);
		if (exist_pdev) {
			put_device(&exist_pdev->dev);
		} else {
			g_rknpu.dyn_pdevs[i] =
				of_platform_device_create(np, NULL, NULL);
		}
		of_node_put(np);
	}

	return platform_driver_register(&rknpu_core_driver);
}

static void __exit rknpu_module_exit(void)
{
	int i;

	if (g_rknpu.misc_registered) {
		misc_deregister(&g_rknpu.miscdev);
		g_rknpu.misc_registered = false;
	}

	platform_driver_unregister(&rknpu_core_driver);

	for (i = 0; i < RKNPU_MAX_CORES; i++) {
		if (g_rknpu.dyn_pdevs[i]) {
			of_device_unregister(g_rknpu.dyn_pdevs[i]);
			g_rknpu.dyn_pdevs[i] = NULL;
		}
	}
}

module_init(rknpu_module_init);
module_exit(rknpu_module_exit);

MODULE_DESCRIPTION("RK3588 RKNPU out-of-tree driver for mainline Linux");
MODULE_AUTHOR("Antigravity");
MODULE_LICENSE("GPL");
